"""ZAC-1 encoder parity: the native zac-encoder-cli against the Python encoder (zac.load_encoder on CUDA).

    uv run python ggml/tests/parity/encoder_parity.py --build ggml/build-cuda --encoder-gguf dist/zac-1-encoder-f32.gguf \
        --teacher-gguf dist/zac-1-ssl-teacher-f32.gguf --audio speech.mp3 [--seconds 66.1] [--checkpoint FILE] [--gpu]

The audio is decoded to 44.1 kHz mono with ffmpeg, tiled to --seconds and peak-limited to 1. Both encoders run the
same independent 20-second lanes with 10 s of left context. Codes are not bit-exact (BF16 rounding decisions of
cuDNN / fused SDPA flip RVQ boundary choices), so the gate is agreement: at least 0.65 over all codes and 0.9 on
codebook 0, measured separately before and after frame 900 (60 s) when the clip is longer. Exits 1 below the gate.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from zac import HOP, SAMPLE_RATE, load_encoder  # noqa: E402

FRAMES_60S = 900


def clip(path: Path, seconds: float) -> np.ndarray:
    raw = subprocess.check_output(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    )
    one = np.frombuffer(raw, np.float32)
    n = int(round(seconds * SAMPLE_RATE))
    audio = np.tile(one, n // one.size + 1)[:n].astype(np.float32)
    return audio / max(1.0, float(np.abs(audio).max()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", type=Path, required=True, help="ggml build directory (holds zac-encoder-cli)")
    parser.add_argument("--encoder-gguf", type=Path, required=True)
    parser.add_argument("--teacher-gguf", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True, help="any audio file ffmpeg reads")
    parser.add_argument("--seconds", type=float, default=66.1)
    parser.add_argument("--checkpoint", type=Path, help="encoder safetensors (default: the release file)")
    parser.add_argument("--gpu", action="store_true", help="run zac-encoder-cli on the GPU backend")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("the Python encoder needs CUDA")

    audio = clip(args.audio, args.seconds)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "audio.npy", audio)
        run = subprocess.run(
            [
                str(args.build / "zac-encoder-cli"),
                str(args.encoder_gguf),
                str(args.teacher_gguf),
                "--audio",
                str(tmp / "audio.npy"),
                str(tmp / "codes.npy"),
                "--gpu" if args.gpu else "--cpu",
            ],
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            sys.exit(f"zac-encoder-cli failed:\n{run.stderr}")
        native = np.load(tmp / "codes.npy")

    encoder = load_encoder("cuda:0", checkpoint=args.checkpoint)
    reference = encoder.encode(torch.from_numpy(audio), SAMPLE_RATE, max_seconds=600).numpy()
    expected = (-(-audio.size // HOP), 10)
    if native.shape != expected or reference.shape != expected:
        print(f"shape mismatch: native {native.shape}, Python {reference.shape}, expected {expected}")
        sys.exit(1)
    equal = native == reference
    parts = [("all", equal)] if equal.shape[0] <= FRAMES_60S else [
        ("<60 s", equal[:FRAMES_60S]),
        (">60 s", equal[FRAMES_60S:]),
    ]
    print(f"{args.seconds:g} s, zac-encoder-cli {'--gpu' if args.gpu else '--cpu'}")
    print("per codebook " + " ".join(f"{v:.3f}" for v in equal.mean(0)))
    ok = True
    for name, part in parts:
        print(f"agreement {name}: {part.mean():.4f} (codebook 0 {part[:, 0].mean():.4f})")
        ok = ok and part.mean() >= 0.65 and part[:, 0].mean() >= 0.9
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
