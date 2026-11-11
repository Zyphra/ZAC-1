"""The ZAC-1 decoder: codes -> latent -> transformer stages 0-8 (15 Hz -> 240 Hz) -> multi-resolution iSTFT head at
44160 Hz -> 44100 Hz.

The latent is the fp32 RVQ lookup of the semantic codebook through the semantic ConvNeXt head plus the nine residual
codebooks. The head is a causal ConvNeXt trunk, a LayerNorm, one Linear per window size to n_fft+2 channels
(log-magnitude and phase), one centered iSTFT per branch at hop 184 = 44160/240 (periodic Hann) and a linear FIR mixer.
"""

import math

import torch
from torch import nn
from torchaudio.transforms import Resample

from zac.layers import WNConv1d
from zac.moss import MossStage, build_stages
from zac.quantizer import DualQuantizer
from zac.semantic import SemanticConvNeXt

KEPT_STAGES = 9  # decoder.0 .. decoder.8
HEAD_FRAME_RATE = 240


class MultiResISTFTHead(nn.Module):
    """Multi-resolution iSTFT head: one causal ConvNeXt trunk, one linear projection per window size, each synthesised
    with its own centered iSTFT at the shared hop, then a linear FIR mixer (fp32, no nonlinearity) combines the branches.

    A long window branch (harmonic resolution: 2208 = 12 x 184 gives 20 Hz bins at 50 ms) sits next to the short one
    (736: 60 Hz bins at 16.7 ms, transients), followed by a learned linear post-filter. Nothing after the synthesis is
    nonlinear, so the output band above the features' content is a linear function of them.

    Warm start: blocks beyond the warm-started ones are zero-initialised at pwconv2 (identity at start); the mixer
    starts as a delta on branch 0 and zeros on the others, so at step 0 the output equals the branch-0 head alone and
    the extra branches are switched in by gradient (zero-init gating)."""

    def __init__(self, dim, depth, hidden, n_ffts, hop, fir_kernel=127, warm_blocks=None):
        super().__init__()
        self.hop = int(hop)
        self.n_ffts = [int(n) for n in n_ffts]
        for n in self.n_ffts:
            if n % self.hop or n // self.hop < 2:
                raise ValueError("every n_fft must be an integer multiple (>= 2) of the hop")
        self.blocks = nn.Sequential(*[SemanticConvNeXt(dim, hidden) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.outs = nn.ModuleList([nn.Linear(dim, n + 2) for n in self.n_ffts])
        for i, n in enumerate(self.n_ffts):
            self.register_buffer(f"window_{i}", torch.hann_window(n, periodic=True), persistent=False)
        if fir_kernel % 2 == 0:
            raise ValueError("fir_kernel must be odd (zero-phase at init)")
        self.mixer = nn.Conv1d(len(self.n_ffts), 1, fir_kernel, padding=(fir_kernel - 1) // 2, bias=False)
        with torch.no_grad():
            self.mixer.weight.zero_()
            self.mixer.weight[0, 0, (fir_kernel - 1) // 2] = 1.0  # branch 0 passes through, other branches gated off
        if warm_blocks is not None:
            for block in list(self.blocks)[int(warm_blocks) :]:
                nn.init.zeros_(block.pwconv2.weight)
                nn.init.zeros_(block.pwconv2.bias)
        self.max_log_magnitude = math.log(100.0)

    def spectra(self, x):
        """x [batch, dim, frames] -> list of (magnitude, phase) [batch, n_fft//2+1, frames] (fp32)."""
        h = self.norm(self.blocks(x).transpose(1, 2))
        out = []
        for lin in self.outs:
            y = lin(h)
            log_magnitude, phase = y.float().chunk(2, dim=-1)
            magnitude = torch.exp(log_magnitude.clamp(max=self.max_log_magnitude))
            out.append((magnitude.transpose(1, 2), phase.transpose(1, 2)))
        return out

    def forward(self, x):
        frames = x.shape[-1]
        branches = []
        with torch.autocast(x.device.type, enabled=False):
            for i, (magnitude, phase) in enumerate(self.spectra(x)):
                spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
                n = self.n_ffts[i]
                branches.append(
                    torch.istft(
                        spec,
                        n,
                        hop_length=self.hop,
                        win_length=n,
                        window=getattr(self, f"window_{i}"),
                        center=True,
                        length=frames * self.hop,
                    )
                )
            stacked = torch.stack(branches, dim=1)  # [batch, branches, samples]
            return self.mixer(stacked.float())  # [batch, 1, samples], linear, fp32


class ISTFTCodec(nn.Module):
    """Codes -> latent -> decoder.0..8 features at 240 Hz -> iSTFT head -> 44160 Hz -> 44100 Hz.

    ``cfg`` is the codec's model config, ``head_cfg`` the head config (``type`` ``"mr_istft"``; ``stage_layers``
    optionally prunes the transformer stages, e.g. ``{"0": [0, 31]}`` keeps two of decoder.0's 32 layers)."""

    def __init__(self, cfg, head_cfg):
        super().__init__()
        stages = cfg["decoder_stages"]
        assert stages[KEPT_STAGES - 1]["type"] == "MossProjectedTransformer"
        assert stages[KEPT_STAGES - 1]["frame_rate"] == HEAD_FRAME_RATE
        assert stages[KEPT_STAGES]["type"] == "PatchedPretransform" and stages[KEPT_STAGES]["patch_size"] == 23
        hop = cfg["internal_sample_rate"] // HEAD_FRAME_RATE
        assert hop * HEAD_FRAME_RATE == cfg["internal_sample_rate"] and cfg["internal_hop_length"] == hop * (
            HEAD_FRAME_RATE // cfg["frame_rate"]
        )
        if head_cfg.get("type") != "mr_istft":
            raise ValueError(f"unsupported head type {head_cfg.get('type')!r} (ZAC-1 decoders use 'mr_istft')")
        kept = list(build_stages(stages[:KEPT_STAGES], False))
        kept[-1].output_proj = nn.Identity()  # the 768->5888 projection only fed the removed patch/conv tail
        self.stage_layers = {int(k): [int(i) for i in v] for k, v in head_cfg.get("stage_layers", {}).items()}
        for si, keep in self.stage_layers.items():
            stage = kept[si]
            assert isinstance(stage, MossStage), f"stage {si} is not a transformer stage"
            assert keep == sorted(set(keep)) and keep and all(0 <= i < len(stage.layers) for i in keep), (si, keep)
            stage.layers = nn.ModuleList(stage.layers[i] for i in keep)
        self.stages = nn.Sequential(*kept)
        self.quantizer = DualQuantizer(cfg)
        branch = cfg["dual_encoding"]
        width, hidden, depth = branch["width"], branch["hidden"], branch["depth"]
        self.semantic_head = nn.Sequential(
            *[SemanticConvNeXt(width, hidden) for _ in range(depth)], WNConv1d(width, 1024, 1)
        )
        self.to_external = Resample(44160, 44100, lowpass_filter_width=16, rolloff=0.99)
        self.cfg = cfg
        self.hop = hop
        self.upsample = HEAD_FRAME_RATE // cfg["frame_rate"]  # 16 head frames per code frame
        self.external_hop = cfg["external_samples_per_frame"]
        self.head = MultiResISTFTHead(
            dim=stages[KEPT_STAGES - 1]["d_model"],
            depth=head_cfg["depth"],
            hidden=head_cfg["hidden"],
            n_ffts=head_cfg["n_ffts"],
            hop=hop,
            fir_kernel=head_cfg.get("fir_kernel", 127),
            warm_blocks=head_cfg.get("warm_blocks"),
        )

    @torch.no_grad()
    def latent(self, codes, counts=None):
        """codes [batch, 10, frames] (long); counts [batch] = residual codebooks kept (1..9), None = all.

        fp32 RVQ lookups, fp32 semantic head. Residual codebooks beyond `counts` are dropped per sample, reproducing
        the training-time codebook dropout augmentation on cached codes."""
        if codes.ndim != 3 or codes.shape[1] != 1 + self.cfg["residual_codebooks"]:
            raise ValueError("Expected codes [batch, 10, frames]")
        q = self.quantizer
        with torch.autocast(codes.device.type, enabled=False):
            q1 = q.semantic_quantizer.from_codes(codes[:, :1])[0]
            z = self.semantic_head(q1).float()
            for i, vq in enumerate(q.quantizer.quantizers):
                z_i = vq.out_proj(vq.decode_code(codes[:, 1 + i]))
                if counts is not None:
                    z_i = z_i * (counts > i).to(z_i.dtype)[:, None, None]
                z = z + z_i
        return z

    def features(self, codes, counts=None):
        """240 Hz features [batch, 768, 16*frames] in bf16 autocast on CUDA (fp32 elsewhere)."""
        x = self.latent(codes, counts)
        with torch.autocast(codes.device.type, dtype=torch.bfloat16, enabled=codes.device.type == "cuda"):
            for stage in self.stages:
                with torch.no_grad():
                    x = stage(x)
        return x

    def forward(self, codes, counts=None, length=None):
        feats = self.features(codes, counts)
        with torch.autocast(codes.device.type, dtype=torch.bfloat16, enabled=codes.device.type == "cuda"):
            internal = self.head(feats)
        with torch.autocast(codes.device.type, enabled=False):
            audio = self.to_external(internal.float())
        expected = codes.shape[-1] * self.external_hop
        audio = audio[..., :expected]
        return audio if length is None else audio[..., :length]
