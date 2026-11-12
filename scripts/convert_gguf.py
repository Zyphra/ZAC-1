"""Convert the ZAC-1 safetensors checkpoints to the GGUFs of the native ggml port (ggml/).

    uv run python scripts/convert_gguf.py decoder --variant {keep2_plain,full_mr} [--checkpoint F] [--outtype {f16,f32}] [-o OUT]
    uv run python scripts/convert_gguf.py encoder [--checkpoint F] [--outtype {f32,f16}] [-o OUT]
    uv run python scripts/convert_gguf.py ssl-teacher [--checkpoint F] [--outtype {f32,bf16,f16}] [-o OUT]

``--checkpoint`` defaults to the release file from ``Zyphra/ZAC-1`` (``ssl-teacher`` reads the ``teacher.*`` group of
the encoder file). Every subcommand loads its weights into the module that runs them (zac.decoder.ISTFTCodec,
zac.encoder_model.EncoderModel, transformers' Wav2Vec2BertModel) with a strict load, so an unknown or missing tensor
is an error, and reads the written file back (architecture, tensor names, shapes and types). ``--help`` on a
subcommand describes its GGUF layout. The architectures (mossdual_istft, mossdual_encoder, w2vbert_ssl) and their keys
are those of the ZONOS3-Mini GGUFs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from release_checks import (  # noqa: E402
    DEC_STEP,
    HEAD_STEP,
    LAYER_KEYS,
    fail,
    validate_head_config,
    validate_model_config,
    validate_teacher_config,
)

from zac.checkpoint import DECODER_FILES, ENCODER_FILE, read_checkpoint, resolve  # noqa: E402

_DTYPES = {np.dtype(np.float32): GGMLQuantizationType.F32, np.dtype(np.float16): GGMLQuantizationType.F16}


class Writer(GGUFWriter):
    """GGUFWriter that records every tensor's ggml shape (ne) and type for the read-back check."""

    def __init__(self, path, arch):
        super().__init__(str(path), arch)
        self.path, self.arch = Path(path), arch
        self.written: dict[str, tuple[tuple[int, ...], GGMLQuantizationType]] = {}

    def add_tensor(self, name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
        super().add_tensor(name, tensor, raw_shape=raw_shape, raw_dtype=raw_dtype, tensor_endianess=tensor_endianess)
        qtype = raw_dtype if raw_dtype is not None else _DTYPES[tensor.dtype]
        self.written[name] = (tuple(int(n) for n in reversed(tensor.shape)), qtype)


def write_gguf(writer: Writer, progress: bool = False) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=progress)
    writer.close()
    reader = GGUFReader(str(writer.path))
    if reader.fields["general.architecture"].contents() != writer.arch:
        sys.exit(f"{writer.path}: general.architecture did not round-trip")
    found = {t.name: t for t in reader.tensors}
    if len(reader.tensors) != len(writer.written) or found.keys() != writer.written.keys():
        sys.exit(f"{writer.path}: tensor names did not round-trip")
    for name, (shape, qtype) in writer.written.items():
        t = found[name]
        if t.tensor_type != qtype:
            sys.exit(f"{writer.path}: {name} has type {t.tensor_type.name}, expected {qtype.name}")
        if tuple(int(n) for n in t.shape) != shape:
            sys.exit(f"{writer.path}: {name} has ne {tuple(int(n) for n in t.shape)}, expected {shape}")
    print(f"{writer.path}: {writer.path.stat().st_size} bytes, {len(found)} tensors, read back", flush=True)


# ===================================================================================================== decoder

DECODER_DOC = """Convert a ZAC-1 decoder (zac-1-decoder-{keep2_plain,full_mr}.safetensors) to a mossdual_istft GGUF.

The model is zac.decoder.ISTFTCodec: the quantizers (RVQ tables folded through their 1x1 out_proj), the semantic
ConvNeXt head, transformer stages 0,2,4,6,8, the multi-resolution iSTFT head and the 44160->44100 resampler kernel.
Weight-normed convs are materialized (w = g * v / ||v||). Transformer, semantic-head and iSTFT-head matrices follow
--outtype; every other tensor is f32 (the Hann windows are stored so the runtime uses torch's exact values)."""

DEC_ARCH = "mossdual_istft"
STAGE_PATCH = [2, 2, 2, 2, 1]  # patches the runtime applies
HEADROOM_DB = 3.0
LAYER_NAMES = [
    "norm1.weight",
    "norm1.bias",
    "qkv.weight",
    "wo.weight",
    "attn_scale",
    "norm2.weight",
    "norm2.bias",
    "ffn1.weight",
    "ffn2.weight",
    "ffn_scale",
]
LAYER_MAT = {"qkv.weight", "wo.weight", "ffn1.weight", "ffn2.weight"}


def wn(g, v):
    """torch weight norm over every dim but 0."""
    return g * v / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1))


def _wn_key(get, sd, prefix):
    if prefix + ".weight_g" in sd:
        return wn(get(prefix + ".weight_g"), get(prefix + ".weight_v"))
    return wn(get(prefix + ".parametrizations.weight.original0"), get(prefix + ".parametrizations.weight.original1"))


def check_codec_step(md):
    if md.get("codec_step") != str(DEC_STEP):
        fail(f"codec_step={md.get('codec_step')}")


def cmd_decoder(args):
    from zac.decoder import ISTFTCodec

    variant = args.variant
    checkpoint = resolve(args.checkpoint, DECODER_FILES[variant])
    out = args.out or Path("dist") / f"zac-1-decoder-{variant}-{args.outtype}.gguf"
    md, groups = read_checkpoint(checkpoint, "decoder")
    if md.get("variant") != variant:
        sys.exit(f"{checkpoint} holds the {md.get('variant')!r} decoder, not {variant!r}")
    check_codec_step(md)
    if md.get("head_step") != str(HEAD_STEP):
        fail(f"head_step={md.get('head_step')}")
    if float(md["headroom_db"]) != HEADROOM_DB:
        fail(f"headroom_db={md['headroom_db']}")
    model_config, head = json.loads(md["model_config"]), json.loads(md["head_config"])
    moss, sh = validate_model_config(model_config)
    layers, want = validate_head_config(head, moss)
    got = {k for k in groups["codec"] if k.startswith(("stages.", "head."))}
    if got != want:
        fail(f"decoder keys differ: missing {sorted(want - got)[:8]}, unexpected {sorted(got - want)[:8]}")
    model = ISTFTCodec(model_config, head)
    model.load_state_dict(groups["codec"], strict=True)
    del groups
    sd = model.state_dict()
    consumed = set()

    def get(key):
        consumed.add(key)
        return sd[key].float()

    out.parent.mkdir(parents=True, exist_ok=True)
    w = Writer(out, DEC_ARCH)
    w.add_string("general.name", f"ZAC-1 decoder ({variant}, head step {HEAD_STEP})")
    for key, val in {
        "sample_rate": 44100,
        "internal_sample_rate": 44160,
        "frame_rate": 15,
        "hop": 2940,
        "internal_hop": 2944,
        "n_codebooks": 10,
        "codebook_size": 4096,
        "latent_dim": 1024,
        "semantic_dim": sh["width"],
        "sem_head.depth": sh["depth"],
        "head.depth": head["depth"],
        "head.hidden": head["hidden"],
        "head.hop": head["hop"],
        "head.fir_kernel": head["fir_kernel"],
        "head.step": HEAD_STEP,
    }.items():
        w.add_uint32(f"{DEC_ARCH}.{key}", val)
    w.add_array(f"{DEC_ARCH}.head.n_fft", [int(n) for n in head["n_ffts"]])
    w.add_float32(f"{DEC_ARCH}.head.max_log_magnitude", math.log(100.0))
    w.add_float32(f"{DEC_ARCH}.headroom_db", HEADROOM_DB)
    w.add_string(f"{DEC_ARCH}.head.variant", variant)
    w.add_float32(f"{DEC_ARCH}.ln_eps", 1e-5)
    w.add_float32(f"{DEC_ARCH}.convnext_eps", 1e-6)
    for key, field in {
        "in_dim": "input_dimension",
        "out_dim": "output_dimension",
        "d_model": "d_model",
        "n_head": "num_heads",
        "n_ff": "dim_feedforward",
        "context": "context_frames",
    }.items():
        vals = [int(s[field]) for s in moss]
        if key == "out_dim":
            vals[4] = 0  # decoder.8.output_proj is an Identity in the iSTFT codec
        w.add_array(f"{DEC_ARCH}.stage.{key}", vals)
    w.add_array(f"{DEC_ARCH}.stage.n_layer", layers)
    w.add_array(f"{DEC_ARCH}.stage.patch", STAGE_PATCH)

    kernel = get("to_external.kernel")  # [new, 1, 2*width + orig]
    new, _, kl = kernel.shape
    orig, width = 736, (kl - 736) // 2
    if new != 735 or kl != 2 * width + orig:
        fail(f"to_external.kernel shape {tuple(kernel.shape)}")
    w.add_uint32(f"{DEC_ARCH}.resample.orig", orig)
    w.add_uint32(f"{DEC_ARCH}.resample.new", new)
    w.add_uint32(f"{DEC_ARCH}.resample.width", width)

    count = 0

    def put(name, t, mat=False):
        nonlocal count
        a = t.detach().cpu().float().numpy()
        if not np.isfinite(a).all():
            raise SystemExit(f"non-finite values in {name}")
        if mat and args.outtype == "f16":
            h = a.astype(np.float16)
            if not np.isfinite(h).all():
                raise SystemExit(f"F16 overflow in {name}")
            w.add_tensor(name, np.ascontiguousarray(h))
        else:
            w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))
        count += 1

    def convnext(o, p):
        put(o + ".dw.weight", get(p + ".dwconv.weight").squeeze(1))
        put(o + ".dw.bias", get(p + ".dwconv.bias"))
        put(o + ".norm.weight", get(p + ".norm.weight"))
        put(o + ".norm.bias", get(p + ".norm.bias"))
        put(o + ".pw1.weight", get(p + ".pwconv1.weight"), mat=True)
        put(o + ".pw1.bias", get(p + ".pwconv1.bias"))
        put(o + ".pw2.weight", get(p + ".pwconv2.weight"), mat=True)
        put(o + ".pw2.bias", get(p + ".pwconv2.bias"))

    with torch.no_grad():
        # RVQ codebooks folded through their 1x1 out_proj: table[id] = codebook[id] @ W^T.
        def table(prefix):
            return (
                get(prefix + ".codebook.weight") @ _wn_key(get, sd, prefix + ".out_proj").squeeze(-1).T,
                get(prefix + ".out_proj.bias"),
            )

        t, b = table("quantizer.semantic_quantizer.quantizers.0")
        put("sem.table", t)
        put("sem.bias", b)
        # res.bias = the nine out_proj biases summed (what a complete frame adds); res.{i}.bias = each one alone, which
        # a decode leaving residual codebook i out of a frame (early partial-frame decode) subtracts.
        total = torch.zeros(1024, dtype=torch.float64)
        for i in range(9):
            t, b = table(f"quantizer.quantizer.quantizers.{i}")
            put(f"res.{i}.table", t)
            put(f"res.{i}.bias", b)
            total += b.double()
        put("res.bias", total.float())
        consumed.update(k for k in sd if ".in_proj." in k and k.startswith("quantizer."))  # encode-only

        for b in range(sh["depth"]):
            convnext(f"sem_head.{b}", f"semantic_head.{b}")
        put("sem_head.out.weight", _wn_key(get, sd, f"semantic_head.{sh['depth']}").squeeze(-1), mat=True)
        put("sem_head.out.bias", get(f"semantic_head.{sh['depth']}.bias"))

        for si, s in enumerate(range(0, 9, 2)):
            p, o = f"stages.{s}", f"stage.{si}"
            put(o + ".in.weight", get(p + ".input_proj.weight"), mat=True)
            if s < 8:
                put(o + ".out.weight", get(p + ".output_proj.weight"), mat=True)
            for layer in range(layers[si]):
                for src, dst in zip(LAYER_KEYS, LAYER_NAMES):
                    put(f"{o}.l{layer}.{dst}", get(f"{p}.layers.{layer}.{src}"), mat=dst in LAYER_MAT)

        for b in range(head["depth"]):
            convnext(f"head.{b}", f"head.blocks.{b}")
        put("head.norm.weight", get("head.norm.weight"))
        put("head.norm.bias", get("head.norm.bias"))
        for i, n in enumerate(head["n_ffts"]):
            ow = get(f"head.outs.{i}.weight")
            if tuple(ow.shape) != (n + 2, 768):
                fail(f"head.outs.{i}.weight shape {tuple(ow.shape)}")
            put(f"head.out.{i}.weight", ow, mat=True)
            put(f"head.out.{i}.bias", get(f"head.outs.{i}.bias"))
        mixer = get("head.mixer.weight")
        if tuple(mixer.shape) != (1, len(head["n_ffts"]), head["fir_kernel"]):
            fail(f"head.mixer.weight shape {tuple(mixer.shape)}")
        put("head.mixer.weight", mixer.squeeze(0))
        for i, n in enumerate(head["n_ffts"]):
            put(f"head.window.{i}", torch.hann_window(n, periodic=True))
        put("resample.kernel", kernel.squeeze(1))

    leftover = sorted(k for k in sd if k not in consumed)
    if leftover:
        raise SystemExit(f"{len(leftover)} ISTFTCodec tensors not mapped, e.g.: {leftover[:8]}")
    write_gguf(w)
    print(f"wrote {count} {DEC_ARCH} tensors to {out}")


# ===================================================================================================== encoder

ENCODER_DOC = """Convert the ZAC-1 encode path (the codec.* group of zac-1-encoder.safetensors, zac.encoder_model.EncoderModel)
to a mossdual_encoder GGUF.

Stores original RVQ codevectors and both projections in F32, never an inverse of folded decode tables. Acoustic
convolutions are tap-major [K,OC,IC], ConvNeXt kernels [K,C], matching the decoder's native repacked graph convention."""

ENC_ARCH = "mossdual_encoder"


def cmd_encoder(args):
    from zac.encoder_model import EncoderModel

    checkpoint = resolve(args.checkpoint, ENCODER_FILE)
    out = args.out or Path("dist") / f"zac-1-encoder-{args.outtype}.gguf"
    md, groups = read_checkpoint(checkpoint, "encoder")
    check_codec_step(md)
    cfg = json.loads(md["model_config"])
    _, sh = validate_model_config(cfg)
    specs = cfg["encoder_stages"]
    if [s["type"] for s in specs] != ["S1CausalWNConv1d", "S1EncoderBlock", "S1EncoderBlock"] + [
        "PatchedPretransform",
        "MossProjectedTransformer",
    ] * 5:
        raise SystemExit("unsupported encoder stages")
    patches = [s["patch_size"] for s in specs[3::2]]
    if patches != [23, 2, 2, 2, 2] or any(s["direction"] != "down" for s in specs[3::2]):
        raise SystemExit("unsupported encoder patching")
    if [s["stride"] for s in specs[1:3]] != [2, 4] or cfg["dual_encoding"] != {
        "width": 768,
        "hidden": 2048,
        "depth": 4,
        "causal": True,
        "window_ratio": 2.0,
        "strategy": "frozen_ssl_to_q1_decoded_semantic_residual",
        "normalization": "unchanged_baseline",
    }:
        raise SystemExit("unsupported S1/semantic encoder configuration")
    moss = specs[4::2]
    model = EncoderModel(cfg)
    model.load_state_dict(groups["codec"], strict=True, assign=True)
    del groups
    sd = model.state_dict()
    consumed = set()

    def get(k):
        consumed.add(k)
        return sd[k].detach().float()

    def wnp(p):
        return _wn_key(get, sd, p)

    out.parent.mkdir(parents=True, exist_ok=True)
    w = Writer(out, ENC_ARCH)
    w.add_string("general.name", "ZAC-1 encoder")
    for key, value in {
        "sample_rate": 44100,
        "internal_sample_rate": 44160,
        "frame_rate": 15,
        "hop": 2940,
        "internal_hop": 2944,
        "n_codebooks": 10,
        "codebook_size": 4096,
        "latent_dim": 1024,
        "semantic_dim": 768,
        "semantic_depth": 4,
        "lane_frames": 300,
        "context_frames": 150,
    }.items():
        w.add_uint32(f"{ENC_ARCH}.{key}", value)
    w.add_float32(f"{ENC_ARCH}.window_ratio", cfg["dual_encoding"]["window_ratio"])
    w.add_float32(f"{ENC_ARCH}.ln_eps", 1e-5)
    w.add_float32(f"{ENC_ARCH}.convnext_eps", 1e-6)
    for key, field in {
        "in_dim": "input_dimension",
        "out_dim": "output_dimension",
        "d_model": "d_model",
        "n_head": "num_heads",
        "n_layer": "num_layers",
        "n_ff": "dim_feedforward",
        "context": "context_frames",
    }.items():
        w.add_array(f"{ENC_ARCH}.stage.{key}", [int(s[field]) for s in moss])
    w.add_array(f"{ENC_ARCH}.stage.patch", patches)
    count = 0

    def put(name, t, mat=False):
        nonlocal count
        a = t.cpu().numpy().astype(np.float16 if mat and args.outtype == "f16" else np.float32)
        if not np.isfinite(a).all():
            raise SystemExit(f"nonfinite/overflow tensor {name}")
        w.add_tensor(name, np.ascontiguousarray(a))
        count += 1

    def conv(o, p):
        # Raw torch [OC,IC,K] -> ggml [IC,OC,K], taps contiguous matrices.
        put(o + ".weight", wnp(p + ".conv").permute(2, 0, 1))
        put(o + ".bias", get(p + ".conv.bias").reshape(-1, 1))

    def snake(o, p):
        alpha = get(p).reshape(-1, 1)
        put(o + ".alpha", alpha)
        put(o + ".inv_alpha", 1 / (alpha + 1e-9))

    def convnext(o, p):
        put(o + ".dw.weight", get(p + ".dwconv.weight").squeeze(1).T)
        put(o + ".dw.bias", get(p + ".dwconv.bias"))
        for name in ["norm.weight", "norm.bias"]:
            put(o + "." + name, get(p + "." + name))
        for src, dst in [("pwconv1", "pw1"), ("pwconv2", "pw2")]:
            put(o + "." + dst + ".weight", get(p + "." + src + ".weight"))
            put(o + "." + dst + ".bias", get(p + "." + src + ".bias"))

    with torch.no_grad():
        kernel = get("to_internal.kernel").squeeze(1)
        new, kl = kernel.shape
        orig = 735
        width = (kl - orig) // 2
        if new != 736 or kl != 2 * width + orig:
            raise SystemExit("unsupported encoder resampling kernel")
        for key, value in {"orig": orig, "new": new, "width": width}.items():
            w.add_uint32(f"{ENC_ARCH}.resample.{key}", value)
        put("resample.kernel", kernel)
        conv("enc.in", "encoder.0")
        for b in range(2):
            p, o = f"encoder.{b + 1}.module.block", f"enc.b{b}"
            for r in range(3):
                rp, ro = f"{p}.{r}.module.block", f"{o}.res{r}"
                snake(ro + ".snake1", rp + ".0.alpha")
                conv(ro + ".conv1", rp + ".1")
                snake(ro + ".snake2", rp + ".2.alpha")
                conv(ro + ".conv2", rp + ".3")
            snake(o + ".snake", p + ".3.alpha")
            conv(o + ".conv", p + ".4")
        for s, spec in enumerate(moss):
            p, o = f"encoder.{4 + 2 * s}", f"stage.{s}"
            put(o + ".in.weight", get(p + ".input_proj.weight"), mat=True)
            put(o + ".out.weight", get(p + ".output_proj.weight"), mat=True)
            for layer in range(spec["num_layers"]):
                lp, lo = f"{p}.layers.{layer}", f"{o}.l{layer}"
                for src, dst in [
                    ("norm1.weight", "norm1.weight"),
                    ("norm1.bias", "norm1.bias"),
                    ("norm2.weight", "norm2.weight"),
                    ("norm2.bias", "norm2.bias"),
                    ("attention_scale", "attn_scale"),
                    ("ffn_scale", "ffn_scale"),
                    ("qkv.weight", "qkv.weight"),
                    ("attention_out.weight", "wo.weight"),
                    ("ffn.0.weight", "ffn1.weight"),
                    ("ffn.3.weight", "ffn2.weight"),
                ]:
                    put(
                        lo + "." + dst,
                        get(lp + "." + src),
                        mat=dst in ["qkv.weight", "wo.weight", "ffn1.weight", "ffn2.weight"],
                    )
        put("sem_enc.in.weight", wnp("semantic_encoder.0").squeeze(-1))
        put("sem_enc.in.bias", get("semantic_encoder.0.bias"))
        for b in range(sh["depth"]):
            convnext(f"sem_enc.{b}", f"semantic_encoder.{b + 1}")
            convnext(f"sem_head.{b}", f"semantic_head.{b}")
        put("sem_head.out.weight", wnp(f"semantic_head.{sh['depth']}").squeeze(-1))
        put("sem_head.out.bias", get(f"semantic_head.{sh['depth']}.bias"))
        for c in range(10):
            p = "quantizer.semantic_quantizer.quantizers.0" if c == 0 else f"quantizer.quantizer.quantizers.{c - 1}"
            for proj in ["in_proj", "out_proj"]:
                put(f"rvq.{c}.{proj}.weight", wnp(p + "." + proj).squeeze(-1))
                put(f"rvq.{c}.{proj}.bias", get(p + "." + proj + ".bias"))
            put(f"rvq.{c}.codebook", get(p + ".codebook.weight"))
    required = ("encoder.", "semantic_encoder.", "semantic_head.", "quantizer.", "to_internal.")
    leftover = sorted(k for k in sd if k.startswith(required) and k not in consumed)
    if leftover:
        raise SystemExit(f"unmapped encoder tensors: {leftover}")
    write_gguf(w)
    print(f"wrote {count} encoder tensors to {out}")


# ===================================================================================================== ssl-teacher

SSL_DOC = """Export the sixteen-layer semantic teacher (the teacher.* group of zac-1-encoder.safetensors: transformers
Wav2Vec2BertModel, Wav2Vec2-BERT 2.0) to a w2vbert_ssl GGUF.

The only non-inference checkpoint tensor, masked_spec_embed, is explicitly consumed and checked: spec augmentation is
disabled in the frozen SemanticTeacher. Frontend tables retain their original float64 precision; resampling
coefficients are the exact default torchaudio.transforms.Resample(44100, 16000) float32 kernel. Use --outtype bf16 for
matrix storage without runtime weight casts; normalization, biases and depthwise/frontend tables keep their opmath
precision."""

SSL_ARCH = "w2vbert_ssl"


def ssl_mappings():
    # Each entry contains GGUF name, model parameter name, exact shape,
    # optional squeezed convolution axes, and whether weights must remain F32.
    result = []

    def tensor(dst, src, shape, squeeze=(), sensitive=False):
        result.append((dst, src, tuple(shape), squeeze, sensitive))

    def pair(dst, src, inp, out):
        tensor(dst + ".weight", src + ".weight", (out, inp))
        tensor(dst + ".bias", src + ".bias", (out,), sensitive=True)

    def norm(dst, src, dim):
        tensor(dst + ".weight", src + ".weight", (dim,), sensitive=True)
        tensor(dst + ".bias", src + ".bias", (dim,), sensitive=True)

    norm("fp.ln", "feature_projection.layer_norm", 160)
    pair("fp.proj", "feature_projection.projection", 160, 1024)
    for i in range(16):
        src, dst = f"encoder.layers.{i}.", f"enc.{i}."
        for j in (1, 2):
            norm(dst + f"ffn{j}_ln", src + f"ffn{j}_layer_norm", 1024)
            pair(dst + f"ffn{j}.up", src + f"ffn{j}.intermediate_dense", 1024, 4096)
            pair(dst + f"ffn{j}.down", src + f"ffn{j}.output_dense", 4096, 1024)
        norm(dst + "attn_ln", src + "self_attn_layer_norm", 1024)
        for a, b in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "out")):
            pair(dst + "attn." + a, src + "self_attn.linear_" + b, 1024, 1024)
        tensor(dst + "attn.dist", src + "self_attn.distance_embedding.weight", (73, 64), sensitive=True)
        norm(dst + "conv_ln", src + "conv_module.layer_norm", 1024)
        tensor(dst + "conv.pw1.weight", src + "conv_module.pointwise_conv1.weight", (2048, 1024, 1), (2,))
        tensor(dst + "conv.dw.weight", src + "conv_module.depthwise_conv.weight", (1024, 1, 31), (1,), True)
        norm(dst + "conv.dw_ln", src + "conv_module.depthwise_layer_norm", 1024)
        tensor(dst + "conv.pw2.weight", src + "conv_module.pointwise_conv2.weight", (1024, 1024, 1), (2,))
        norm(dst + "final_ln", src + "final_layer_norm", 1024)
    return result


def cmd_ssl_teacher(args):
    import torchaudio
    from gguf import GGMLQuantizationType as Q
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel

    checkpoint = resolve(args.checkpoint, ENCODER_FILE)
    out = args.out or Path("dist") / f"zac-1-ssl-teacher-{args.outtype}.gguf"
    md, groups = read_checkpoint(checkpoint, "encoder")
    config = json.loads(md["teacher_config"])
    validate_teacher_config(config)
    # fp32 model, strict load of the stored bf16 tensors upcast to fp32 (exact): the values of
    # Wav2Vec2BertModel.from_pretrained(dtype=torch.float32) on the bf16 file.
    model = Wav2Vec2BertModel(Wav2Vec2BertConfig(**config)).float()
    model.load_state_dict({k: v.float() for k, v in groups["teacher"].items()}, strict=True)
    del groups
    sd = model.state_dict()
    mapping = ssl_mappings()
    expected = {src: shape for _, src, shape, _, _ in mapping}
    expected["masked_spec_embed"] = (1024,)
    keys = set(sd)
    if keys != set(expected):
        raise ValueError(
            f"teacher tensor schema mismatch: missing={sorted(set(expected) - keys)}, unexpected={sorted(keys - set(expected))}"
        )
    for key, shape in expected.items():
        if tuple(sd[key].shape) != shape:
            raise ValueError(f"wrong shape for {key}: {tuple(sd[key].shape)}, expected {shape}")
    if not torch.isfinite(sd["masked_spec_embed"]).all():
        raise ValueError("nonfinite training-only masked_spec_embed")
    out.parent.mkdir(parents=True, exist_ok=True)
    w = Writer(out, SSL_ARCH)
    w.add_string("general.name", "ZAC-1 semantic teacher (Wav2Vec2-BERT 2.0, 16 layers)")
    meta = {
        "sample_rate": 16000,
        "max_seconds": 60,
        "n_layer": 16,
        "n_embd": 1024,
        "n_head": 16,
        "n_ff": 4096,
        "rel_left": 64,
        "rel_right": 8,
        "conv_kernel": 31,
        "n_mels": 80,
        "frame_length": 400,
        "hop_length": 160,
        "fft_length": 512,
        "stride": 2,
    }
    for key, value in meta.items():
        w.add_uint32(f"{SSL_ARCH}.{key}", value)
    for key, value in {"eps": 1e-5, "preemphasis": 0.97, "mel_floor": 1.192092955078125e-07}.items():
        w.add_float32(f"{SSL_ARCH}.{key}", value)
    w.add_float32(f"{SSL_ARCH}.padding_value", 1.0)
    fe = SeamlessM4TFeatureExtractor(feature_size=80, sampling_rate=16000, num_mel_bins=80, padding_value=1, stride=2)
    for name, value in (("frontend.window", fe.window), ("frontend.mel_filters", fe.mel_filters)):
        w.add_tensor(name, np.ascontiguousarray(value, dtype=np.float64), raw_dtype=Q.F64)
    resampler = torchaudio.transforms.Resample(44100, 16000)
    w.add_tensor("frontend.resample_44100", resampler.kernel[:, 0, :].contiguous().numpy())
    consumed = {"masked_spec_embed"}
    for dst, src, _, squeeze, sensitive in mapping:
        value = sd[src].float().numpy()
        if squeeze:
            value = np.squeeze(value, axis=squeeze)
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite tensor: {src}")
        if args.outtype == "f16" and value.ndim == 2 and not sensitive:
            if np.max(np.abs(value)) >= 65504:
                raise ValueError(f"f16 overflow: {src}")
            value = value.astype(np.float16)
        if args.outtype == "bf16" and value.ndim == 2 and not sensitive:
            bits = torch.from_numpy(np.ascontiguousarray(value)).to(torch.bfloat16).view(torch.uint16).numpy()
            w.add_tensor(dst, bits, raw_dtype=Q.BF16)
        else:
            w.add_tensor(dst, np.ascontiguousarray(value))
        consumed.add(src)
    if consumed != keys:
        raise ValueError(f"unconsumed teacher tensors: {sorted(keys - consumed)}")
    write_gguf(w, progress=True)
    print(
        f"consumed {len(consumed)} checkpoint tensors; exported {len(mapping)} inference tensors and 3 frontend tables -> {out}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, doc, fn):
        p = sub.add_parser(name, help=doc.splitlines()[0], description=doc,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.set_defaults(fn=fn)
        p.add_argument("-o", "--out", type=Path, help="output path (default: under dist/)")
        return p

    p = command("decoder", DECODER_DOC, cmd_decoder)
    p.add_argument("--checkpoint", type=Path, help="decoder safetensors (default: the release file)")
    p.add_argument("--variant", choices=tuple(DECODER_FILES), required=True)
    p.add_argument("--outtype", choices=["f16", "f32"], default="f16")

    p = command("encoder", ENCODER_DOC, cmd_encoder)
    p.add_argument("--checkpoint", type=Path, help=f"encoder safetensors (default: {ENCODER_FILE})")
    p.add_argument("--outtype", choices=["f32", "f16"], default="f32")

    p = command("ssl-teacher", SSL_DOC, cmd_ssl_teacher)
    p.add_argument("--checkpoint", type=Path, help=f"encoder safetensors (default: {ENCODER_FILE})")
    p.add_argument("--outtype", choices=("f32", "bf16", "f16"), default="f32")

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
