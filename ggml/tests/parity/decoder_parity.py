"""ZAC-1 decoder parity: the native zac-cli against the Python decoder (zac.load_decoder, CPU fp32, raw level).

    uv run python ggml/tests/parity/decoder_parity.py --build ggml/build --gguf dist/zac-1-decoder-keep2_plain-f16.gguf \
        [--checkpoint dist/zac-1-decoder-keep2_plain.safetensors] [--codes codes.npy] [--variant keep2_plain] [--gpu]

Codes default to a seeded random [150, 10] matrix. Prints cos and max_abs; exits 1 when the lengths differ or
cos < 0.999.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from zac import DECODER_FILES, load_decoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", type=Path, required=True, help="ggml build directory (holds zac-cli)")
    parser.add_argument("--gguf", type=Path, required=True, help="decoder GGUF")
    parser.add_argument("--checkpoint", type=Path, help="decoder safetensors (default: the release file)")
    parser.add_argument("--codes", type=Path, help="ordinary codes [frames, 10] .npy (default: seeded random)")
    parser.add_argument("--variant", choices=tuple(DECODER_FILES), default="keep2_plain")
    parser.add_argument("--gpu", action="store_true", help="run zac-cli on the GPU backend")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if args.codes is None:
            codes = np.random.default_rng(0).integers(0, 4096, (150, 10)).astype("<i4")
        else:
            codes = np.rint(np.load(args.codes)).astype("<i4")
        np.save(tmp / "codes.npy", codes)
        run = subprocess.run(
            [
                str(args.build / "zac-cli"),
                str(args.gguf),
                str(tmp / "codes.npy"),
                str(tmp / "out.npy"),
                "--headroom-db",
                "0",
                "--gpu" if args.gpu else "--cpu",
            ],
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            sys.exit(f"zac-cli failed:\n{run.stderr}")
        native = np.load(tmp / "out.npy").astype(np.float64)

    python = load_decoder(args.variant, device="cpu", checkpoint=args.checkpoint).decode(codes).numpy()
    python = python.astype(np.float64)
    print(f"variant {args.variant}, {codes.shape[0]} frames, zac-cli {'--gpu' if args.gpu else '--cpu'}")
    if native.shape != python.shape:
        print(f"length mismatch: Python {python.size}, C++ {native.size}")
        sys.exit(1)
    cos = float(python @ native / (np.linalg.norm(python) * np.linalg.norm(native)))
    print(f"cos={cos:.8f} max_abs={np.max(np.abs(python - native)):.6g}")
    if cos < 0.999:
        sys.exit(1)


if __name__ == "__main__":
    main()
