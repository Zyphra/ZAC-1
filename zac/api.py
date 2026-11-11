"""Load the ZAC-1 decoders and encoder from ``zac-1/1`` safetensors checkpoints (default: the Hugging Face release)."""

from __future__ import annotations

import json
import os

import torch

from zac.checkpoint import (
    CODEBOOK_SIZE,
    CODEBOOKS,
    DECODER_FILES,
    DEFAULT_DECODER,
    ENCODER_FILE,
    HOP,
    read_checkpoint,
    resolve,
)
from zac.decoder import ISTFTCodec
from zac.encoder import ZacEncoder
from zac.encoder_model import EncoderModel
from zac.semantic import SemanticTeacher
from zac.stream import ChunkGraphs, ISTFTStream


class ZacDecoder:
    """A ZAC-1 decoder variant (``keep2_plain`` or ``full_mr``) on one device.

    ``decode`` returns audio at the raw decoder level; multiply by ``output_gain()`` (the variant's ``headroom_db``
    of attenuation) for playback level."""

    def __init__(self, model: ISTFTCodec, variant: str, headroom_db: float, device: torch.device):
        self.model = model
        self.variant = variant
        self.headroom_db = float(headroom_db)
        self.device = torch.device(device)
        self.graphs: ChunkGraphs | None = None

    def decode(self, codes) -> torch.Tensor:
        """Integer codes [frames, 10] or [batch, frames, 10] in [0, 4096) -> float32 CPU audio [frames * 2940] or
        [batch, frames * 2940] at 44.1 kHz (fp32 with TF32 off)."""
        codes = torch.as_tensor(codes)
        single = codes.ndim == 2
        if single:
            codes = codes[None]
        if (
            codes.ndim != 3
            or codes.shape[-1] != CODEBOOKS
            or codes.dtype.is_floating_point
            or codes.dtype.is_complex
            or codes.dtype == torch.bool
            or bool(((codes < 0) | (codes >= CODEBOOK_SIZE)).any())
        ):
            raise ValueError("Invalid ZAC-1 codes")
        model = self.model
        device = next(model.parameters()).device
        packed = codes.permute(0, 2, 1).to(device=device, dtype=torch.int64)
        expected = packed.shape[-1] * HOP
        prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.inference_mode():
                audio = model(packed)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
        if audio.shape != (packed.shape[0], 1, expected):
            raise RuntimeError(f"Codec output shape {tuple(audio.shape)}; expected {expected} samples")
        if not torch.isfinite(audio).all():
            raise RuntimeError("Nonfinite codec audio")
        audio = audio.float().squeeze(1).cpu()
        return audio[0] if single else audio

    def stream(self, graphs: ChunkGraphs | None = None) -> ISTFTStream:
        """A fresh stateful stream decoder; it replays ``graphs`` (default: the ones ``capture_stream_graphs`` made)."""
        return ISTFTStream(self.model, graphs=graphs if graphs is not None else self.graphs)

    def capture_stream_graphs(self, chunks) -> None:
        """CUDA-graph the chunk shapes of a stream decoding ``chunks`` frames per call from the start (bit-identical to
        the eager decode; a no-op off CUDA). Later ``stream()`` calls replay them."""
        if next(self.model.parameters()).device.type != "cuda":
            return
        if self.graphs is None:
            self.graphs = ChunkGraphs()
        self.graphs.capture(self.model, chunks)

    def output_gain(self) -> float:
        return 10 ** (-self.headroom_db / 20)


def load_decoder(
    variant: str = DEFAULT_DECODER,
    device: str | torch.device | None = None,
    checkpoint: str | os.PathLike | None = None,
) -> ZacDecoder:
    """The ``variant`` decoder from ``checkpoint`` (default: its release file from ``Zyphra/ZAC-1``); ``device``
    defaults to CUDA when available."""
    if variant not in DECODER_FILES:
        raise ValueError(f"unknown ZAC-1 decoder {variant!r} (decoders: {', '.join(DECODER_FILES)})")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    path = resolve(checkpoint, DECODER_FILES[variant])
    md, groups = read_checkpoint(path, "decoder")
    if md.get("variant") != variant:
        raise ValueError(f"{path} holds the {md.get('variant')!r} decoder, not {variant!r}")
    model = ISTFTCodec(json.loads(md["model_config"]), json.loads(md["head_config"]))
    model.load_state_dict(groups["codec"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return ZacDecoder(model, variant, float(md["headroom_db"]), device)


def load_encoder(
    device: str | torch.device | None = None,
    checkpoint: str | os.PathLike | None = None,
    cuda_graph: bool = True,
) -> ZacEncoder:
    """The ZAC-1 encoder from ``checkpoint`` (default: ``zac-1-encoder.safetensors`` from ``Zyphra/ZAC-1``) on a CUDA
    ``device`` (default ``cuda:0``)."""
    device = torch.device(device or "cuda:0")
    if device.type != "cuda":
        raise RuntimeError("ZAC-1 encoding requires CUDA; use the ggml port (zac-encoder-cli) on CPU")
    md, groups = read_checkpoint(resolve(checkpoint, ENCODER_FILE), "encoder")
    model = EncoderModel(json.loads(md["model_config"]))
    model.load_state_dict(groups["codec"], strict=True, assign=True)
    teacher = SemanticTeacher(json.loads(md["teacher_config"]), groups["teacher"])
    return ZacEncoder(model, teacher, device, cuda_graph=cuda_graph)
