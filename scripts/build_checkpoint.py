"""Build the three ZAC-1 safetensors files from the ZONOS3-Mini release weights.

    uv run python scripts/build_checkpoint.py --generator autoencoder/generator.pt --heads autoencoder/heads \
        --teacher autoencoder/teacher [--out-dir dist] [--push Zyphra/ZAC-1]

The inputs are ``autoencoder/`` of ``Zyphra/ZONOS3-Mini`` at revision 24f39592cf5521d675d96db41fc379af3ad369be.
Outputs (format ``zac-1/1``):

- ``zac-1-encoder.safetensors``: ``codec.*`` = generator.pt's encoder, semantic encoder/head, quantizers and
  44100 -> 44160 Hz resampler (fp32); ``teacher.*`` = the 517 Wav2Vec2-BERT 2.0 tensors of teacher/model.safetensors
  (bf16, as stored).
- ``zac-1-decoder-{keep2_plain,full_mr}.safetensors``: ``codec.*`` = generator.pt's quantizers, semantic head and
  44160 -> 44100 Hz resampler plus the head checkpoint's ``trainable`` dict (decoder stages 0-8 and the iSTFT head; it
  overrides every generator decoder weight, so those are not stored).

Every file's modules load strictly before anything is written. ``--push`` uploads the files to a Hugging Face model
repo (created public if it does not exist).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from release_checks import (  # noqa: E402
    DEC_STEP,
    HEAD_STEP,
    VARIANTS,
    check_generator_sha256,
    find_head_files,
    sha256_file,
    validate_generator,
    validate_head,
    validate_teacher_config,
)

from zac.checkpoint import DECODER_FILES, ENCODER_FILE, FORMAT, write_checkpoint  # noqa: E402
from zac.decoder import ISTFTCodec  # noqa: E402
from zac.encoder_model import EncoderModel  # noqa: E402
from zac.semantic import SemanticTeacher  # noqa: E402

SOURCE = "Zyphra/ZONOS3-Mini@24f39592cf5521d675d96db41fc379af3ad369be"
ENCODER_PREFIXES = ("encoder.", "semantic_encoder.", "semantic_head.", "quantizer.", "to_internal.")
DECODER_PREFIXES = ("quantizer.", "semantic_head.", "to_external.")
TEACHER_TENSORS = 517
HEADROOM_DB = "3.0"


def report(path: Path) -> None:
    print(f"{path}  {path.stat().st_size} bytes  sha256 {sha256_file(path)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generator", type=Path, required=True, help="autoencoder/generator.pt")
    parser.add_argument("--heads", type=Path, required=True, help="autoencoder/heads (keep2_plain/ and full_mr/)")
    parser.add_argument("--teacher", type=Path, required=True, help="autoencoder/teacher")
    parser.add_argument("--out-dir", type=Path, default=Path("dist"))
    parser.add_argument("--push", metavar="REPO_ID", help="upload to this Hugging Face model repo")
    args = parser.parse_args()

    from safetensors.torch import load_file

    check_generator_sha256(args.generator)
    state = torch.load(args.generator, map_location="cpu", mmap=True, weights_only=True)
    gen_cfg, moss, _ = validate_generator(state)
    generator = state["generator"]

    teacher_config = json.loads((args.teacher / "config.json").read_text())
    validate_teacher_config(teacher_config)
    teacher_state = load_file(str(args.teacher / "model.safetensors"))
    if len(teacher_state) != TEACHER_TENSORS:
        sys.exit(f"expected {TEACHER_TENSORS} teacher tensors, found {len(teacher_state)}")

    # Strict loads of every file's modules before anything is written.
    encoder_codec = {k: v for k, v in generator.items() if k.startswith(ENCODER_PREFIXES)}
    EncoderModel(gen_cfg).load_state_dict(encoder_codec, strict=True)
    SemanticTeacher(teacher_config, teacher_state)
    decoders = {}
    for variant in ("keep2_plain", "full_mr"):
        head_dir = args.heads / variant
        name, cfg_path, ckpt_path = find_head_files(head_dir)
        assert name == variant and name in VARIANTS
        hc = json.loads(cfg_path.read_text())
        ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
        validate_head(hc, ck, gen_cfg, moss, cfg_path.name)
        codec = {k: v for k, v in generator.items() if k.startswith(DECODER_PREFIXES)} | dict(ck["trainable"])
        ISTFTCodec(hc["model"], hc["head"]).load_state_dict(codec, strict=True)
        decoders[variant] = (hc, codec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    path = args.out_dir / ENCODER_FILE
    write_checkpoint(
        path,
        {"codec": encoder_codec, "teacher": teacher_state},
        {
            "format": FORMAT,
            "kind": "encoder",
            "model_config": json.dumps(gen_cfg, sort_keys=True),
            "teacher_config": json.dumps(teacher_config, sort_keys=True),
            "teacher_layer": "16",
            "codec_step": str(DEC_STEP),
            "source": f"{SOURCE} autoencoder/generator.pt + autoencoder/teacher",
        },
    )
    report(path)
    written.append(path)
    for variant, (hc, codec) in decoders.items():
        path = args.out_dir / DECODER_FILES[variant]
        write_checkpoint(
            path,
            {"codec": codec},
            {
                "format": FORMAT,
                "kind": "decoder",
                "variant": variant,
                "model_config": json.dumps(hc["model"], sort_keys=True),
                "head_config": json.dumps(hc["head"], sort_keys=True),
                "headroom_db": HEADROOM_DB,
                "head_step": str(HEAD_STEP),
                "codec_step": str(DEC_STEP),
                "source": f"{SOURCE} autoencoder/generator.pt + autoencoder/heads/{variant}",
            },
        )
        report(path)
        written.append(path)

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, repo_type="model", private=False, exist_ok=True)
        for path in written:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=args.push,
                commit_message=f"Add {path.name}",
            )
            print(f"uploaded to https://huggingface.co/{args.push}/blob/main/{path.name}", flush=True)


if __name__ == "__main__":
    main()
