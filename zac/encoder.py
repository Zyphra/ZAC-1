"""Waveform -> ZAC-1 codes on CUDA: the native SSL teacher -> semantic codebook 0, acoustic residual RVQ -> codebooks 1-9.

Each independent lane retains both external-rate teacher audio and full-file resampled codec audio. Packing external
hop samples into the second channel is lossless and only aligns the two sample grids for the batch scheduler. The
teacher sees 20-second lanes, including the same 10-second real left context as the acoustic encoder. It retains native
bidirectional attention within a lane. Preprocessing (``prepare_audio``): mono, resample to 44.1 kHz with torchaudio,
peak normalize only above 1.0, keep whole codec frames up to the input limit.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from zac.batcher import IndependentBatcher
from zac.checkpoint import CODEBOOK_SIZE, CODEBOOKS, HOP, SAMPLE_RATE
from zac.encoder_model import EncoderModel
from zac.resample import resample_exact
from zac.semantic import SemanticTeacher, extract_ssl

MAX_SECONDS = 600.0
BATCH_SIZE = 1
CHUNK_SECONDS = 20.0
CONTEXT_SECONDS = 10.0


def prepare_audio(waveform: torch.Tensor, sample_rate: int, max_seconds: float = MAX_SECONDS) -> torch.Tensor:
    """Waveform [samples] or [channels, samples] at ``sample_rate`` -> mono float32 [1, N] at 44.1 kHz: channels
    averaged, resampled with torchaudio, divided by its peak only when that exceeds 1, cut to the whole 2940-sample
    hops within ``max_seconds`` (in (0, 600]). ValueError for a nonfinite or empty input or one shorter than 0.5 s."""
    if not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError(f"Codec input limit must be in (0, {MAX_SECONDS:g}] seconds")
    if waveform.ndim == 1:
        waveform = waveform[None]
    if waveform.ndim != 2 or not waveform.shape[-1] or not torch.isfinite(waveform).all():
        raise ValueError("Expected a finite, nonempty reference waveform")
    waveform = waveform.detach().cpu().float().mean(0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        import torchaudio.functional as AF

        waveform = AF.resample(waveform, int(sample_rate), SAMPLE_RATE)
    if waveform.shape[-1] < SAMPLE_RATE // 2:
        raise ValueError("Reference shorter than 0.5 s")
    peak = float(waveform.abs().amax())
    if peak > 1.0:
        waveform = waveform / peak
    maximum = math.floor(max_seconds * SAMPLE_RATE / HOP) * HOP
    return waveform[:, :maximum].contiguous()


def _cuda_graph(function, warmup):
    """One fixed CUDA shape of ``function`` as a CUDA graph (normal PyTorch kernels, so discrete outputs stay
    bit-identical); returns ``replay(*args)``, which copies its inputs into the captured ones."""
    warmup_args = warmup if isinstance(warmup, tuple) else (warmup,)
    device = warmup_args[0].device
    with torch.inference_mode():
        function(*warmup_args)
        torch.cuda.synchronize(device)
        for _ in range(2):
            output = function(*warmup_args)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = function(*warmup_args)
        torch.cuda.synchronize(device)

    def replay(*args):
        if len(args) != len(warmup_args):
            raise ValueError("fixed CUDA graph received the wrong argument count")
        for static, current in zip(warmup_args, args):
            if static.shape != current.shape or static.dtype != current.dtype:
                raise ValueError("fixed CUDA graph received an incompatible tensor")
            static.copy_(current)
        graph.replay()
        return output

    return replay


class ZacEncoder:
    """The ZAC-1 encoder on one CUDA device: 20 s lanes with 10 s of real left context, batch 1, the acoustic encoder
    as a CUDA graph (``cuda_graph``), the teacher and the semantic modules eager. Warm-up asserts that the graph and the
    eager path give identical codes."""

    prepare_audio = staticmethod(prepare_audio)

    def __init__(self, model: EncoderModel, teacher: SemanticTeacher, device, cuda_graph: bool = True):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("ZAC-1 encoding requires CUDA; use the ggml port (zac-encoder-cli) on CPU")
        torch.cuda.set_device(self.device)
        cfg = model.cfg
        self.cfg = cfg
        self.codebook_sizes = [cfg["semantic_codebook_size"]] + [cfg["residual_codebook_size"]] * 9
        assert cfg["residual_codebooks"] == 9 and cfg["frame_rate"] == 15 and cfg["dual_encoding"]
        assert (
            cfg["sample_rate"],
            cfg["internal_sample_rate"],
            cfg["internal_hop_length"],
            cfg["external_samples_per_frame"],
        ) == (44100, 44160, 2944, 2940)
        model.eval().requires_grad_(False)
        self.resample = model.to_internal.cpu()
        del model.to_internal
        self.model = model.to(self.device)
        self.teacher = teacher.to(self.device).eval().requires_grad_(False)
        self.sample_rate = 44100
        self.feature_dim = 10
        self.batch_size = BATCH_SIZE
        self.chunk_seconds = CHUNK_SECONDS
        self.context_seconds = CONTEXT_SECONDS
        self.packer = IndependentBatcher(
            torch=torch,
            device=self.device,
            sample_rate=44160,
            hop_length=2944,
            feature_dim=10,
            prepare=self._prepare,
            run=self._run,
            batch_size=self.batch_size,
            chunk_seconds=self.chunk_seconds,
            continuation_context_seconds=self.context_seconds,
            window=32,
            output_dtype=np.int16,
            label="zac-encoder",
            input_dtype=torch.float32,
            input_channels=2,
        )
        self.optimized = None
        with torch.inference_mode():
            g = torch.Generator(device=self.device).manual_seed(728)
            warmup = (
                torch.randn((self.batch_size, 1, self.packer.chunk_samples), generator=g, device=self.device) * 0.05
            )
            expected = self._encoder(warmup).clone()
            self.optimized = _cuda_graph(self._encoder, warmup) if cuda_graph else None
            assert torch.equal(self.optimized(warmup) if self.optimized else self._encoder(warmup), expected), (
                "Graph altered acoustic latent"
            )
            for scale in [0.05, 0.1]:
                packed = (
                    torch.randn((self.batch_size, 2, self.packer.chunk_samples), generator=g, device=self.device)
                    * scale
                )
                assert torch.equal(self._run(packed), self._codes(packed)), "Graph/native dual tokens disagree"

    def _external(self, packed):
        b = packed.shape[0]
        return packed[:, 1:2].reshape(b, 1, -1, 2944)[..., :2940].reshape(b, 1, -1)

    def _encoder(self, audio):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.encoder(audio)

    def _dual(self, packed, optimized):
        # Uses the exact CPU feature preprocessor and frozen teacher from training.
        audio = packed[:, 0:1].contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ssl = extract_ssl(self._external(packed), self.teacher)
            latent = self.optimized(audio) if optimized and self.optimized is not None else self.model.encoder(audio)
            result = self.model.quantize_dual(latent, ssl)
        return result["codes"].transpose(1, 2)

    def _codes(self, packed):
        return self._dual(packed, False)

    def _run(self, packed):
        return self._dual(packed, True)

    def _prepare(self, audio, sample_rate):
        if sample_rate != 44100:
            raise ValueError("Expected 44100 Hz")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[0] != 1 or not audio.shape[-1]:
            raise ValueError("Expected nonempty mono audio")
        if not np.isfinite(audio).all():
            raise ValueError("Nonfinite source audio")
        with torch.inference_mode():
            wave = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
            frames = (wave.shape[-1] + 2939) // 2940
            internal = resample_exact(wave, self.resample)
            internal = F.pad(internal, (0, frames * 2944 - internal.shape[-1]))
            external = F.pad(wave, (0, frames * 2940 - wave.shape[-1])).reshape(1, 1, frames, 2940)
            external = F.pad(external, (0, 4)).reshape(1, 1, frames * 2944)
            return torch.cat([internal, external], dim=1)

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor, sample_rate: int, max_seconds: float = MAX_SECONDS) -> torch.Tensor:
        """Waveform [samples] or [channels, samples] -> int32 codes [ceil(N / 2940), 10] of the prepared audio
        (``prepare_audio``: N samples at 44.1 kHz, at most ``max_seconds``)."""
        waveform = prepare_audio(waveform, sample_rate, max_seconds)
        index = self.device.index or 0
        previous_device = torch.cuda.current_device()
        previous_stream = torch.cuda.current_stream(previous_device)
        torch.cuda.set_device(index)
        try:
            sample = SimpleNamespace(audio=waveform.numpy(), sample_rate=SAMPLE_RATE, error=None)
            sample, codes = next(iter(self.packer.encode_stream([sample])))
        finally:
            torch.cuda.set_device(previous_device)
            torch.cuda.set_stream(previous_stream)
        if sample.error is not None:
            raise RuntimeError(f"ZAC-1 encoding failed: {sample.error}")
        codes = np.asarray(codes)
        expected = math.ceil(waveform.shape[-1] / HOP)
        if codes.ndim != 2 or codes.shape != (expected, CODEBOOKS):
            raise RuntimeError(f"ZAC-1 encoder returned codes of shape {codes.shape}; expected {(expected, CODEBOOKS)}")
        if codes.min() < 0 or codes.max() >= CODEBOOK_SIZE:
            raise RuntimeError("ZAC-1 encoder returned out-of-range codes")
        return torch.from_numpy(codes.astype(np.int32)).contiguous()
