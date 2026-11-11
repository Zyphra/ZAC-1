"""The dual bottleneck: one semantic codebook on the 768-wide semantic branch, nine residual codebooks on the 1024-wide
acoustic latent (4096 entries of dimension 8 each)."""

from torch import nn

from zac.layers import ResidualVectorQuantize


class DualQuantizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.semantic_quantizer = ResidualVectorQuantize(
            input_dim=cfg["dual_encoding"]["width"],
            n_codebooks=1,
            codebook_size=cfg["semantic_codebook_size"],
            codebook_dim=cfg["codebook_dim"],
        )
        self.quantizer = ResidualVectorQuantize(
            1024, cfg["residual_codebooks"], cfg["residual_codebook_size"], cfg["codebook_dim"]
        )
