"""A tiny random ZAC-1 decoder with the release decoder's layout, for tests that need no checkpoint."""

import json

import torch

from zac.checkpoint import FORMAT, write_checkpoint
from zac.decoder import ISTFTCodec

HOP = 2940
TINY_HEAD = {"type": "mr_istft", "depth": 2, "hidden": 64, "n_ffts": [736, 2208], "fir_kernel": 127}


def tiny_config():
    def moss(inp, out, ctx, rate):
        return {
            "type": "MossProjectedTransformer",
            "input_dimension": inp,
            "output_dimension": out,
            "d_model": 32,
            "num_heads": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "context_frames": ctx,
            "frame_rate": rate,
            "layer_scale": 1.0,
            "attention_dropout": 0.0,
            "ffn_dropout": 0.0,
        }

    def up(size):
        return {"type": "PatchedPretransform", "patch_size": size, "direction": "up"}

    stages = [
        moss(1024, 64, 6, 15),
        up(2),
        moss(32, 64, 12, 30),
        up(2),
        moss(32, 64, 24, 60),
        up(2),
        moss(32, 64, 48, 120),
        up(2),
        moss(32, 0, 96, 240),
        up(23),
    ]
    return {
        "decoder_stages": stages,
        "encoder_stages": [],
        "dual_encoding": {"width": 32, "hidden": 64, "depth": 4},
        "residual_codebooks": 9,
        "residual_codebook_size": 4096,
        "semantic_codebook_size": 4096,
        "codebook_dim": 8,
        "quantizer_dropout": 0.0,
        "checkpoint_transformers": False,
        "internal_sample_rate": 44160,
        "frame_rate": 15,
        "internal_hop_length": 2944,
        "external_samples_per_frame": HOP,
    }


def tiny_decoder() -> ISTFTCodec:
    """Five MOSS stages of one layer, the MR iSTFT head with two branches and a mixer through every tap (seeded, eval
    mode)."""
    torch.manual_seed(0)
    m = ISTFTCodec(tiny_config(), TINY_HEAD).eval()
    with torch.no_grad():
        m.head.mixer.weight.normal_(0, 0.05)  # both branches through every tap (the init is a delta on branch 0)
    return m


def tiny_checkpoint(path, variant="keep2_plain", **overrides):
    """Write ``tiny_decoder()`` as a ``zac-1/1`` decoder file; ``overrides`` replace metadata entries."""
    metadata = {
        "format": FORMAT,
        "kind": "decoder",
        "variant": variant,
        "model_config": json.dumps(tiny_config(), sort_keys=True),
        "head_config": json.dumps(TINY_HEAD, sort_keys=True),
        "headroom_db": "3.0",
        "head_step": "50000",
        "codec_step": "600000",
        "source": "tests/tiny.py",
    } | overrides
    write_checkpoint(path, {"codec": tiny_decoder().state_dict()}, metadata)
    return path
