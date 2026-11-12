#!/usr/bin/env python3
"""Emit a tiny mossdual_istft (ZAC-1 decoder) GGUF for CI: real rates, patches, contexts, code ids
and iSTFT head geometry (hop 184, n_fft 736/2208, FIR 127), shrunk widths. Tensor names/shapes mirror
`scripts/convert_gguf.py decoder`.

Usage:  python scripts/make-dummy-zac.py -o dummy-zac.gguf [--codes-out dummy-zac-codes.npy] [--variant full_mr]
"""

from __future__ import annotations

import argparse

import numpy as np
from gguf import GGUFWriter

ARCH = "mossdual_istft"
LATENT, SEM, SEM_DEPTH, SEM_HIDDEN = 16, 16, 1, 32
IN_DIM = [16, 8, 8, 8, 8]
OUT_DIM = [16, 16, 16, 16, 0]
D_MODEL, N_HEAD, N_LAYER, N_FF = 32, 1, 1, 32  # head_dim 32: a Metal flash-attention size
CONTEXT = [150, 300, 480, 480, 480]
PATCH = [2, 2, 2, 2, 1]
HEAD_DEPTH, HEAD_HIDDEN, HEAD_HOP, FIR, N_FFT, HEAD_STEP = 1, 32, 184, 127, [736, 2208], 50000
RS_ORIG, RS_NEW, RS_WIDTH = 736, 735, 17


def write_gguf(writer: GGUFWriter, progress: bool = False) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=progress)
    writer.close()



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--codes-out", help="also write deterministic delayed codes [40,10] and PATH.eos.npy = [31, 0]")
    ap.add_argument(
        "--variant",
        default="full_mr",
        choices=["full_mr", "keep2_plain"],
        help="mossdual_istft.head.variant: the decoder id the servers report (same weights either way)",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    w = GGUFWriter(args.out, ARCH)
    w.add_string("general.name", "mossdual_istft dummy")
    for key, val in {
        "sample_rate": 44100,
        "internal_sample_rate": 44160,
        "frame_rate": 15,
        "hop": 2940,
        "internal_hop": 2944,
        "n_codebooks": 10,
        "codebook_size": 4096,
        "latent_dim": LATENT,
        "semantic_dim": SEM,
        "sem_head.depth": SEM_DEPTH,
        "resample.orig": RS_ORIG,
        "resample.new": RS_NEW,
        "resample.width": RS_WIDTH,
    }.items():
        w.add_uint32(f"{ARCH}.{key}", val)
    w.add_float32(f"{ARCH}.ln_eps", 1e-5)
    w.add_float32(f"{ARCH}.convnext_eps", 1e-6)
    for key, val in {
        "head.depth": HEAD_DEPTH,
        "head.hidden": HEAD_HIDDEN,
        "head.hop": HEAD_HOP,
        "head.fir_kernel": FIR,
        "head.step": HEAD_STEP,
    }.items():
        w.add_uint32(f"{ARCH}.{key}", val)
    w.add_array(f"{ARCH}.head.n_fft", N_FFT)
    w.add_float32(f"{ARCH}.head.max_log_magnitude", float(np.log(100.0)))
    w.add_float32(f"{ARCH}.headroom_db", 3.0)
    w.add_string(f"{ARCH}.head.variant", args.variant)
    for key, vals in {
        "in_dim": IN_DIM,
        "out_dim": OUT_DIM,
        "d_model": [D_MODEL] * 5,
        "n_head": [N_HEAD] * 5,
        "n_layer": [N_LAYER] * 5,
        "n_ff": [N_FF] * 5,
        "context": CONTEXT,
        "patch": PATCH,
    }.items():
        w.add_array(f"{ARCH}.stage.{key}", vals)
    count = 0

    def put(name, arr):
        nonlocal count
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
        count += 1

    def rnd(*shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(n):
        return np.ones(n, dtype=np.float32)

    put("sem.table", rnd(4096, SEM, scale=1.0))
    put("sem.bias", rnd(SEM))
    for i in range(9):
        put(f"res.{i}.table", rnd(4096, LATENT, scale=1.0))
    total = rnd(LATENT)
    put("res.bias", total)
    # res.{i}.bias: the per-codebook biases adding up to res.bias (early partial-frame decode subtracts a missing
    # codebook's); their own generator leaves every other tensor as before.
    biases = np.random.default_rng(1).standard_normal((9, LATENT)) * 0.5
    biases[-1] = total.astype(np.float64) - biases[:-1].sum(0)
    for i in range(9):
        put(f"res.{i}.bias", biases[i])
    for b in range(SEM_DEPTH):
        p = f"sem_head.{b}"
        put(p + ".dw.weight", rnd(SEM, 7))
        put(p + ".dw.bias", rnd(SEM))
        put(p + ".norm.weight", ones(SEM))
        put(p + ".norm.bias", rnd(SEM))
        put(p + ".pw1.weight", rnd(SEM_HIDDEN, SEM))
        put(p + ".pw1.bias", rnd(SEM_HIDDEN))
        put(p + ".pw2.weight", rnd(SEM, SEM_HIDDEN))
        put(p + ".pw2.bias", rnd(SEM))
    put("sem_head.out.weight", rnd(LATENT, SEM))
    put("sem_head.out.bias", rnd(LATENT))
    for s in range(5):
        p = f"stage.{s}"
        put(p + ".in.weight", rnd(D_MODEL, IN_DIM[s], scale=0.3))
        if OUT_DIM[s] > 0:
            put(p + ".out.weight", rnd(OUT_DIM[s], D_MODEL, scale=0.3))
        for l in range(N_LAYER):
            q = f"{p}.l{l}"
            put(q + ".norm1.weight", ones(D_MODEL))
            put(q + ".norm1.bias", rnd(D_MODEL))
            put(q + ".qkv.weight", rnd(3 * D_MODEL, D_MODEL, scale=0.3))
            put(q + ".wo.weight", rnd(D_MODEL, D_MODEL, scale=0.3))
            put(q + ".attn_scale", ones(D_MODEL) * 0.1)
            put(q + ".norm2.weight", ones(D_MODEL))
            put(q + ".norm2.bias", rnd(D_MODEL))
            put(q + ".ffn1.weight", rnd(N_FF, D_MODEL, scale=0.3))
            put(q + ".ffn2.weight", rnd(D_MODEL, N_FF, scale=0.3))
            put(q + ".ffn_scale", ones(D_MODEL) * 0.1)
    for b in range(HEAD_DEPTH):
        p = f"head.{b}"
        put(p + ".dw.weight", rnd(D_MODEL, 7))
        put(p + ".dw.bias", rnd(D_MODEL))
        put(p + ".norm.weight", ones(D_MODEL))
        put(p + ".norm.bias", rnd(D_MODEL))
        put(p + ".pw1.weight", rnd(HEAD_HIDDEN, D_MODEL))
        put(p + ".pw1.bias", rnd(HEAD_HIDDEN))
        put(p + ".pw2.weight", rnd(D_MODEL, HEAD_HIDDEN))
        put(p + ".pw2.bias", rnd(D_MODEL))
    put("head.norm.weight", ones(D_MODEL))
    put("head.norm.bias", rnd(D_MODEL))
    for i, n in enumerate(N_FFT):
        put(f"head.out.{i}.weight", rnd(n + 2, D_MODEL, scale=0.3))
        put(f"head.out.{i}.bias", rnd(n + 2, scale=0.1))
    mixer = 0.01 * rng.standard_normal((len(N_FFT), FIR)).astype(np.float32)
    mixer[0, (FIR - 1) // 2] = 1.0
    put("head.mixer.weight", mixer)
    for i, n in enumerate(N_FFT):  # torch.hann_window(n, periodic=True)
        put(f"head.window.{i}", (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32))
    put("resample.kernel", rnd(RS_NEW, 2 * RS_WIDTH + RS_ORIG))
    write_gguf(w, progress=False)
    print(f"wrote {count} mossdual_istft dummy tensors to {args.out}")
    if args.codes_out:
        codes = np.random.default_rng(0).integers(0, 4096, size=(40, 10)).astype(np.float32)
        np.save(args.codes_out, codes)
        np.save(args.codes_out + ".eos.npy", np.array([31, 0], dtype=np.float32))


if __name__ == "__main__":
    main()
