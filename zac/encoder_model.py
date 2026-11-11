"""The ZAC-1 encoder modules: the causal S1 conv front end + MOSS transformer stages (44160 Hz -> 15 Hz acoustic latent),
the semantic ConvNeXt encoder and head on pooled SSL features, and the dual quantizer.

Codebook 0 quantizes the semantic branch (the teacher's layer-16 features averaged onto the codec frames); codebooks
1-9 quantize the acoustic latent minus the decoded semantic prediction."""

import torch
from torch import nn
from torchaudio.transforms import Resample

from zac.layers import WNConv1d
from zac.moss import build_stages
from zac.quantizer import DualQuantizer
from zac.semantic import SemanticConvNeXt, average_ssl_frames


class EncoderModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = build_stages(cfg["encoder_stages"], cfg["checkpoint_transformers"])
        branch = cfg["dual_encoding"]
        width, hidden, depth = branch["width"], branch["hidden"], branch["depth"]
        self.semantic_encoder = nn.Sequential(
            WNConv1d(1024, width, 1), *[SemanticConvNeXt(width, hidden) for _ in range(depth)]
        )
        self.semantic_head = nn.Sequential(
            *[SemanticConvNeXt(width, hidden) for _ in range(depth)], WNConv1d(width, 1024, 1)
        )
        self.quantizer = DualQuantizer(cfg)
        self.to_internal = Resample(44100, 44160, lowpass_filter_width=16, rolloff=0.99)

    def quantize_dual(self, acoustic, ssl_features, n_quantizers=None):
        if ssl_features.ndim != 3 or ssl_features.shape[0] != acoustic.shape[0] or ssl_features.shape[2] != 1024:
            raise ValueError("Expected SSL features [batch,frames,1024]")
        pooled = average_ssl_frames(ssl_features, acoustic.shape[-1], self.cfg["dual_encoding"]["window_ratio"])
        semantic = self.semantic_encoder(pooled)
        with torch.autocast(acoustic.device.type, enabled=False):
            q1, c1, _, commitment1, codebook1 = self.quantizer.semantic_quantizer(semantic.float())
        semantic = self.semantic_head(q1)
        with torch.autocast(acoustic.device.type, enabled=False):
            residual = acoustic.float() - semantic.float()
            qr, cr, _, commitmentr, codebookr = self.quantizer.quantizer(residual, n_quantizers=n_quantizers)
            z = semantic.float() + qr
        return {
            "z": z,
            "codes": torch.cat([c1, cr], dim=1),
            "semantic_prediction": semantic,
            "commitment": commitment1 + commitmentr,
            "codebook": codebook1 + codebookr,
        }
