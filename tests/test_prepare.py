import pytest
import torch
import torchaudio.functional as AF

from zac import HOP, SAMPLE_RATE, prepare_audio


def tone(seconds, rate=SAMPLE_RATE, amplitude=0.5, channels=1):
    t = torch.arange(int(seconds * rate)) / rate
    return (amplitude * torch.sin(2 * torch.pi * 220 * t)).repeat(channels, 1)


@pytest.mark.parametrize("max_seconds", [0, -1, 600.5])
def test_limit_outside_range_is_rejected(max_seconds):
    with pytest.raises(ValueError, match="limit"):
        prepare_audio(tone(1.0), SAMPLE_RATE, max_seconds)


def test_clip_shorter_than_half_a_second_is_rejected():
    with pytest.raises(ValueError, match="0.5 s"):
        prepare_audio(tone(0.49), SAMPLE_RATE)


def test_stereo_is_averaged():
    left, right = tone(1.0, amplitude=0.2)[0], tone(1.0, amplitude=0.6)[0]
    out = prepare_audio(torch.stack([left, right]), SAMPLE_RATE)
    assert out.shape[0] == 1
    torch.testing.assert_close(out[0], ((left + right) / 2)[: out.shape[-1]])


def test_48k_is_resampled_to_44k1():
    wav = tone(2.0, rate=48000)
    out = prepare_audio(wav, 48000)
    ref = AF.resample(wav, 48000, SAMPLE_RATE)
    assert out.shape[-1] == ref.shape[-1] // HOP * HOP
    torch.testing.assert_close(out, ref[:, : out.shape[-1]])


def test_only_whole_hops_within_the_limit_are_kept():
    out = prepare_audio(tone(10.01), SAMPLE_RATE, max_seconds=10)
    assert out.shape == (1, 441000 // HOP * HOP)


def test_peak_normalized_only_above_one():
    quiet = tone(1.0, amplitude=0.9)
    torch.testing.assert_close(prepare_audio(quiet, SAMPLE_RATE), quiet[:, : SAMPLE_RATE // HOP * HOP])
    loud = tone(1.0, amplitude=2.0)
    out = prepare_audio(loud, SAMPLE_RATE)
    assert float(out.abs().amax()) == pytest.approx(1.0)
    torch.testing.assert_close(out, (loud / loud.abs().amax())[:, : out.shape[-1]])
