import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tiny import HOP, tiny_checkpoint, tiny_decoder

from zac import load_decoder
from zac.checkpoint import read_checkpoint


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tiny_checkpoint(tmp_path_factory.mktemp("ckpt") / "tiny.safetensors")


@pytest.fixture(scope="module")
def codes():
    return np.random.default_rng(0).integers(0, 4096, (12, 10))


def test_loaded_decoder_equals_the_model_it_was_written_from(checkpoint, codes):
    decoder = load_decoder(checkpoint=checkpoint, device="cpu")
    out = decoder.decode(codes)
    with torch.inference_mode():
        ref = tiny_decoder()(torch.as_tensor(codes.T[None], dtype=torch.int64))[0, 0]
    assert out.shape == (12 * HOP,) and out.dtype == torch.float32
    assert torch.equal(out, ref)
    batch = decoder.decode(np.stack([codes, codes]))
    assert batch.shape == (2, 12 * HOP)
    assert decoder.output_gain() == pytest.approx(10 ** (-3 / 20))


def rewrite(src, dst, metadata=None, extra=None):
    with safe_open(str(src), framework="pt") as handle:
        md = dict(handle.metadata())
    tensors = load_file(str(src)) | (extra or {})
    save_file(tensors, str(dst), metadata=md | (metadata or {}))
    return dst


def test_other_format_version_is_rejected(checkpoint, tmp_path):
    bad = rewrite(checkpoint, tmp_path / "v0.safetensors", {"format": "zac-1/0"})
    with pytest.raises(ValueError, match="zac-1/1"):
        load_decoder(checkpoint=bad, device="cpu")


def test_kind_mismatch_is_rejected(checkpoint):
    with pytest.raises(ValueError, match="expected kind 'encoder'"):
        read_checkpoint(checkpoint, "encoder")


def test_unknown_tensor_group_is_rejected(checkpoint, tmp_path):
    bad = rewrite(checkpoint, tmp_path / "extra.safetensors", extra={"foo.x": torch.zeros(1)})
    with pytest.raises(ValueError, match="foo.x"):
        read_checkpoint(bad, "decoder")


def test_variant_mismatch_is_rejected(checkpoint):
    with pytest.raises(ValueError, match="keep2_plain"):
        load_decoder("full_mr", checkpoint=checkpoint, device="cpu")


def test_out_of_range_codes_are_rejected(checkpoint, codes):
    decoder = load_decoder(checkpoint=checkpoint, device="cpu")
    bad = codes.copy()
    bad[3, 5] = 4096
    with pytest.raises(ValueError, match="Invalid ZAC-1 codes"):
        decoder.decode(bad)
    with pytest.raises(ValueError, match="Invalid ZAC-1 codes"):
        decoder.decode(codes[:, :9])
