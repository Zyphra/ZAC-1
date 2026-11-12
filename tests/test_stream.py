"""The stateful ISTFTStream decode equals the whole ISTFTCodec decode (tiny random model, CPU fp32); forks and CUDA-graph
replays (on a GPU) equal fresh eager streams bit for bit."""

import numpy as np
import pytest
import torch
from tiny import HOP, tiny_decoder

from zac.stream import ChunkGraphs, ISTFTStream
from zac.stream_host import ResamplerStream

FRAMES = 40


@pytest.fixture(scope="module")
def model():
    return tiny_decoder()


@pytest.fixture(scope="module")
def codes():
    return np.random.default_rng(0).integers(0, 4096, (FRAMES, 10))


@pytest.fixture(scope="module")
def whole(model, codes):
    with torch.inference_mode():
        return model(torch.as_tensor(codes.T[None], dtype=torch.int64))[0, 0].numpy()


@pytest.mark.parametrize("sizes", [[1, 2, 4, 8, 15, 10], [40]])
def test_stream_equals_whole_decode(model, codes, whole, sizes):
    stream, parts, at = ISTFTStream(model), [], 0
    for n in sizes:
        parts.append(stream.decode(codes[at : at + n]))
        at += n
    out = np.concatenate(parts + [stream.flush()])
    assert len(out) == len(whole) == FRAMES * HOP
    np.testing.assert_allclose(out, whole, atol=1e-5, rtol=0)


def test_stream_emits_only_final_samples(model, codes, whole):
    stream, out = ISTFTStream(model), np.zeros(0, np.float32)
    for f in range(1, 12):
        out = np.concatenate([out, stream.decode(codes[f - 1 : f])])
        assert len(out) == HOP * f - HOP // 2
        np.testing.assert_allclose(out, whole[: len(out)], atol=1e-5, rtol=0)


def decode_in(stream, codes, sizes):
    parts, at = [], 0
    for n in sizes:
        parts.append(stream.decode(codes[at : at + n]))
        at += n
    return np.concatenate(parts + [stream.decode(codes[at:]), stream.flush()])


def test_forks_continue_independently(model, codes):
    """Two forks of a stream that decoded 5 frames continue with different codes; each equals a fresh stream over its
    whole sequence, bit for bit (the twins share no mutable state)."""
    other = np.concatenate([codes[:5], np.random.default_rng(2).integers(0, 4096, (FRAMES - 5, 10))])
    base = ISTFTStream(model)
    head = base.decode(codes[:5])
    a, b = base.fork(), base.fork()
    out_b = b.decode(other[5:20])
    out_a = np.concatenate([head, a.decode(codes[5:20]), a.decode(codes[20:]), a.flush()])
    out_b = np.concatenate([head, out_b, b.decode(other[20:]), b.flush()])
    assert out_a.tobytes() == decode_in(ISTFTStream(model), codes, [5, 15]).tobytes()
    assert out_b.tobytes() == decode_in(ISTFTStream(model), other, [5, 15]).tobytes()


def test_missing_residual_codes_are_left_out(model, codes):
    """A residual code of -1 leaves that codebook out of its frame: codebooks 6.. missing everywhere decode like a codec
    whose codebooks 6.. project to zero, and one frame's missing codes change nothing before that frame (causal)."""
    zeroed = tiny_decoder()
    with torch.no_grad():
        for vq in zeroed.quantizer.quantizer.quantizers[5:]:
            for name, p in vq.out_proj.named_parameters():
                if name != "weight_v":  # weight norm: a zero gain zeroes the weight (zero v would be 0/0)
                    p.zero_()
    missing = codes.copy()
    missing[:, 6:] = -1
    np.testing.assert_allclose(
        decode_in(ISTFTStream(model), missing, [4, 8]), decode_in(ISTFTStream(zeroed), codes, [4, 8]), atol=1e-6, rtol=0
    )
    one = codes.copy()
    one[20, 3:] = -1
    full, partial = decode_in(ISTFTStream(model), codes, [20]), decode_in(ISTFTStream(model), one, [20])
    head = 20 * HOP - HOP // 2  # the samples final before frame 20
    assert full[:head].tobytes() == partial[:head].tobytes() and not np.allclose(full[head:], partial[head:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_graph_replay_equals_eager_decode(model, codes):
    """Captured chunk shapes replay bit-identically to the eager CUDA decode (bf16 trunk), uncaptured shapes fall back
    to eager, and streams interleaving on the shared graphs keep their own state."""
    gpu = tiny_decoder().cuda()
    graphs = ChunkGraphs()
    graphs.capture(gpu, [1, 2, 4, 8])
    other = np.random.default_rng(3).integers(0, 4096, (FRAMES, 10))
    for sizes in ([1, 2, 4, 8], [1, 2, 5, 3]):
        eager = decode_in(ISTFTStream(gpu), codes, sizes)
        assert decode_in(ISTFTStream(gpu, graphs=graphs), codes, sizes).tobytes() == eager.tobytes()
    streams = [(ISTFTStream(gpu, graphs=graphs), codes, []), (ISTFTStream(gpu, graphs=graphs), other, [])]
    at = 0
    for n in (1, 2, 4, 8):
        for stream, frames, parts in streams:
            parts.append(stream.decode(frames[at : at + n]))
        at += n
    for stream, frames, parts in streams:
        out = np.concatenate(parts + [stream.decode(frames[at:]), stream.flush()])
        assert out.tobytes() == decode_in(ISTFTStream(gpu), frames, [1, 2, 4, 8]).tobytes()


def test_resampler_stream_equals_torchaudio(model):
    x = np.random.default_rng(1).standard_normal(20000).astype(np.float32)
    rs = model.to_external
    stream = ResamplerStream(rs.kernel[:, 0].numpy(), rs.orig_freq // rs.gcd, rs.new_freq // rs.gcd, rs.width)
    parts, at = [], 0
    for n in (100, 2944, 5000, len(x)):
        parts.append(stream.push(x[at : at + n]))
        at += n
    out = np.concatenate(parts + [stream.flush()])
    with torch.inference_mode():
        ref = rs(torch.from_numpy(x)).numpy()
    assert len(out) == len(ref)
    np.testing.assert_allclose(out, ref, atol=1e-6, rtol=0)
