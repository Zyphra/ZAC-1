"""The ZAC-1 safetensors checkpoint format (``zac-1/1``) and where the release files live.

Each file carries ``format`` and ``kind`` (``encoder`` or ``decoder``) metadata plus the JSON configs its modules are
built from. Tensors are grouped by their first name component: ``codec.*`` (the codec modules' state dict, keys
unchanged) and, in the encoder file only, ``teacher.*`` (the Wav2Vec2-BERT 2.0 teacher's tensors)."""

from __future__ import annotations

import os

import torch

FORMAT = "zac-1/1"
HF_REPO = "Zyphra/ZAC-1"
ENCODER_FILE = "zac-1-encoder.safetensors"
DECODER_FILES = {
    "keep2_plain": "zac-1-decoder-keep2_plain.safetensors",
    "full_mr": "zac-1-decoder-full_mr.safetensors",
}
DEFAULT_DECODER = "keep2_plain"
SAMPLE_RATE = 44100
HOP = 2940
FRAME_RATE = 15
CODEBOOKS = 10
CODEBOOK_SIZE = 4096

_GROUPS = {"encoder": ("codec", "teacher"), "decoder": ("codec",)}


def read_checkpoint(path: str | os.PathLike, kind: str) -> tuple[dict, dict[str, dict[str, torch.Tensor]]]:
    """A ``zac-1/1`` file of ``kind`` -> (metadata, {group: {name: tensor}}); ValueError for another format or kind
    or a tensor outside the kind's groups."""
    from safetensors import safe_open

    groups: dict[str, dict[str, torch.Tensor]] = {prefix: {} for prefix in _GROUPS[kind]}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != FORMAT:
            raise ValueError(
                f"{path} is not a {FORMAT} checkpoint (format {metadata.get('format')!r}); "
                "build one with scripts/build_checkpoint.py"
            )
        if metadata.get("kind") != kind:
            raise ValueError(f"{path} is a ZAC-1 {metadata.get('kind')!r} checkpoint; expected kind {kind!r}")
        for name in handle.keys():
            prefix, _, rest = name.partition(".")
            if prefix not in groups or not rest:
                raise ValueError(f"unexpected tensor {name!r} in {path} (groups: {', '.join(groups)})")
            groups[prefix][rest] = handle.get_tensor(name)
    return metadata, groups


def write_checkpoint(
    path: str | os.PathLike, groups: dict[str, dict[str, torch.Tensor]], metadata: dict[str, str]
) -> None:
    """Write ``groups`` as ``<group>.<name>`` tensors with string ``metadata`` (include ``format`` and ``kind``)."""
    from safetensors.torch import save_file

    if not all(isinstance(value, str) for value in metadata.values()):
        raise TypeError("checkpoint metadata values must be strings")
    tensors = {f"{prefix}.{name}": t.contiguous() for prefix, group in groups.items() for name, t in group.items()}
    save_file(tensors, str(path), metadata=metadata)


def resolve(checkpoint: str | os.PathLike | None, filename: str) -> str:
    """``checkpoint`` when given, else ``filename`` downloaded from the ``HF_REPO`` model repo (cached)."""
    if checkpoint is not None:
        return str(checkpoint)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO, filename=filename)
