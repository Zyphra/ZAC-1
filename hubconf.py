"""torch.hub entry points: torch.hub.load("Zyphra/ZAC-1", "zac_decoder" | "zac_encoder", trust_repo=True)."""

dependencies = ["torch", "torchaudio", "transformers", "safetensors", "huggingface_hub", "numpy"]

from zac import load_decoder as _load_decoder
from zac import load_encoder as _load_encoder


def zac_decoder(variant="keep2_plain", device=None, checkpoint=None):
    """A ZAC-1 decoder (``keep2_plain``, the default, or ``full_mr``) as a ``ZacDecoder``.

    Downloads ``zac-1-decoder-<variant>.safetensors`` from the Hugging Face repo ``Zyphra/ZAC-1`` unless
    ``checkpoint`` names a local file. ``device`` defaults to CUDA when available.

        decoder = torch.hub.load("Zyphra/ZAC-1", "zac_decoder", trust_repo=True)
        audio = decoder.decode(codes) * decoder.output_gain()  # codes [frames, 10] -> [frames * 2940] at 44.1 kHz
    """
    return _load_decoder(variant=variant, device=device, checkpoint=checkpoint)


def zac_encoder(device=None, checkpoint=None, cuda_graph=True):
    """The ZAC-1 encoder as a ``ZacEncoder`` (CUDA only; ``device`` defaults to ``cuda:0``).

    Downloads ``zac-1-encoder.safetensors`` from the Hugging Face repo ``Zyphra/ZAC-1`` unless ``checkpoint`` names a
    local file.

        encoder = torch.hub.load("Zyphra/ZAC-1", "zac_encoder", trust_repo=True)
        codes = encoder.encode(wav, 44100)  # int32 [ceil(N / 2940), 10]
    """
    return _load_encoder(device=device, checkpoint=checkpoint, cuda_graph=cuda_graph)
