"""Checks on the ZONOS3-Mini release weights that ZAC-1 is built from (``Zyphra/ZONOS3-Mini`` ``autoencoder/``):
generator.pt (codec step 600000), the two decoder heads (step 50000) and the Wav2Vec2-BERT 2.0 teacher.

Every failure exits through ``sys.exit(message)``. Shared by ``build_checkpoint.py`` and ``convert_gguf.py``."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

GENERATOR_SHA256 = "75a587a3eda9ca3d345f0c305756a7ff8619b01685c261b08ae243c69c2caf9a"
DEC_STEP = 600000
HEAD_STEP = 50000
PATCHES = [2, 2, 2, 2, 23]  # generator patches (the last, x23, fed the removed conv tail)
VARIANTS = ("full_mr", "keep2_plain")
HEAD_CHECKS = {
    "type": "mr_istft",
    "depth": 8,
    "hidden": 2048,
    "hop": 184,
    "frame_rate": 240,
    "n_ffts": [736, 2208],
    "fir_kernel": 127,
    "trainable_stages": list(range(9)),
}
LAYER_KEYS = [
    "norm1.weight",
    "norm1.bias",
    "qkv.weight",
    "attention_out.weight",
    "attention_scale",
    "norm2.weight",
    "norm2.bias",
    "ffn.0.weight",
    "ffn.3.weight",
    "ffn_scale",
]
SSL_EXPECTED_CONFIG = {
    "model_type": "wav2vec2-bert",
    "num_hidden_layers": 16,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "feature_projection_input_dim": 160,
    "conv_depthwise_kernel_size": 31,
    "left_max_position_embeddings": 64,
    "right_max_position_embeddings": 8,
    "position_embeddings_type": "relative_key",
    "hidden_act": "swish",
    "layer_norm_eps": 1e-5,
    "add_adapter": False,
    "apply_spec_augment": False,
    "use_intermediate_ffn_before_adapter": False,
}


def fail(msg):
    sys.exit(f"unsupported ZAC-1 release checkpoint: {msg}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_generator_sha256(path: Path) -> None:
    actual = sha256_file(path)
    if actual != GENERATOR_SHA256:
        fail(f"{path} sha256 {actual}, expected {GENERATOR_SHA256}")


def validate_model_config(cfg):
    """The codec model config (generator.pt's ``config.model``) -> (decoder MOSS stage specs, semantic head spec)."""
    for key, want in {
        "architecture": "s1_semantic_rvq_moss_v2_hybrid",
        "sample_rate": 44100,
        "internal_sample_rate": 44160,
        "frame_rate": 15,
        "internal_hop_length": 2944,
        "external_samples_per_frame": 2940,
    }.items():
        if cfg.get(key) != want:
            fail(f"{key}={cfg.get(key)}")
    bn = cfg["bottleneck"]
    for key, want in {
        "type": "dual_encoding_semantic_residual_rvq",
        "semantic_codebooks": 1,
        "residual_codebooks": 9,
        "semantic_codebook_size": 4096,
        "residual_codebook_size": 4096,
        "codebook_dim": 8,
    }.items():
        if bn.get(key) != want:
            fail(f"bottleneck.{key}={bn.get(key)}")
    sh = bn["semantic_head"]
    if (sh.get("type"), sh.get("causal"), sh.get("width"), sh.get("out_channels")) != (
        "ConvNeXtDecoder",
        True,
        768,
        1024,
    ):
        fail(f"semantic_head={sh}")
    stages = cfg["decoder_stages"]
    kinds = [s["type"] for s in stages]
    want_kinds = ["MossProjectedTransformer", "PatchedPretransform"] * 5 + ["S1DecoderBlock"] * 2 + ["S1WaveformHead"]
    if kinds != want_kinds:
        fail(f"decoder stage types {kinds}")
    moss = stages[0:10:2]
    for s in moss:
        if (s["causal"], s["norm"], s["positional_embedding"], s["max_period"], s["gating"], s["activation"]) != (
            True,
            "layer_norm",
            "rope",
            10000,
            "none",
            "gelu",
        ):
            fail(f"moss stage {s}")
    patches = [s["patch_size"] for s in stages[1:10:2]]
    if patches != PATCHES or any(s["direction"] != "up" for s in stages[1:10:2]):
        fail(f"patches {patches}")
    return moss, sh


def validate_generator(state):
    """generator.pt's loaded dict -> (model config, decoder MOSS stage specs, semantic head spec)."""
    if state.get("step") != DEC_STEP:
        fail(f"step={state.get('step')}")
    cfg = state["config"]["model"]
    moss, sh = validate_model_config(cfg)
    return cfg, moss, sh


def find_head_files(head_dir: Path):
    variant = head_dir.name
    if variant not in VARIANTS:
        fail(f"head directory name '{variant}' is not one of {VARIANTS}")
    configs = sorted(head_dir.glob("config_*.json"))
    if len(configs) != 1:
        fail(f"{head_dir} must contain exactly one config_*.json (found {[c.name for c in configs]})")
    ckpt = head_dir / f"step_{HEAD_STEP:07d}.pt"
    if not ckpt.is_file():
        fail(f"missing {ckpt}")
    return variant, configs[0], ckpt


def validate_head_config(head, moss):
    """A head config -> (per decoder MOSS stage layer counts, the trainable state's expected keys)."""
    for key, want in HEAD_CHECKS.items():
        if head.get(key) != want:
            fail(f"head.{key}={head.get(key)}")
    layers = []
    for s in range(0, 9, 2):
        if "stage_layers" in head and str(s) in head["stage_layers"]:
            n = len(head["stage_layers"][str(s)])
        else:
            n = int(moss[s // 2]["num_layers"])
        layers.append(n)
    want = set()
    for s, n in zip(range(0, 9, 2), layers):
        want.add(f"stages.{s}.input_proj.weight")
        if s < 8:
            want.add(f"stages.{s}.output_proj.weight")
        for layer in range(n):
            want.update(f"stages.{s}.layers.{layer}.{k}" for k in LAYER_KEYS)
    for b in range(head["depth"]):
        want.update(
            f"head.blocks.{b}.{k}"
            for k in (
                "dwconv.weight",
                "dwconv.bias",
                "norm.weight",
                "norm.bias",
                "pwconv1.weight",
                "pwconv1.bias",
                "pwconv2.weight",
                "pwconv2.bias",
            )
        )
    want.update(
        [
            "head.norm.weight",
            "head.norm.bias",
            "head.outs.0.weight",
            "head.outs.0.bias",
            "head.outs.1.weight",
            "head.outs.1.bias",
            "head.mixer.weight",
        ]
    )
    return layers, want


def validate_head(hc, ck, gen_cfg, moss, cfg_name):
    """A head's config and checkpoint against generator.pt's config -> (head config, layer counts)."""
    if ck.get("step") != HEAD_STEP:
        fail(f"head checkpoint step={ck.get('step')}")
    if hc.get("model") != gen_cfg:
        fail(f"{cfg_name}: model config differs from generator.pt's")
    head = hc["head"]
    layers, want = validate_head_config(head, moss)
    got = set(ck["trainable"])
    if got != want:
        fail(f"head checkpoint keys differ: missing {sorted(want - got)[:8]}, unexpected {sorted(got - want)[:8]}")
    return head, layers


def validate_teacher_config(config):
    for key, expected in SSL_EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            fail(f"teacher configuration {key}: {config.get(key)!r}, expected {expected!r}")
    if config.get("architectures") != ["Wav2Vec2BertModel"]:
        fail("teacher must be the base Wav2Vec2BertModel, not a task head")
