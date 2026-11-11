"""The semantic branch: the causal ConvNeXt block, Mimi-style SSL frame averaging, the frozen Wav2Vec2-BERT 2.0 teacher
(first 16 layers) and its exact W2V-BERT feature preprocessing."""

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio import transforms as T


class SemanticConvNeXt(nn.Module):
    """Causal form of DualCodec's depthwise-conv/MLP residual block."""

    def __init__(self, width, hidden):
        super().__init__()
        self.dwconv = nn.Conv1d(width, width, 7, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pwconv1 = nn.Linear(width, hidden)
        self.pwconv2 = nn.Linear(hidden, width)

    def forward(self, x):
        h = self.dwconv(F.pad(x, (6, 0))).transpose(1, 2)
        h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
        return x + h.transpose(1, 2)


def average_ssl_frames(target, student_frames, window_ratio=2.0):
    """Noncausal box averages, including all overlapping SSL frame cells.

    Mimi uses kernel=8, stride=4 for 50 -> 12.5 Hz. Here a student interval
    occupies target_frames/student_frames SSL frame cells, and the window is
    window_ratio intervals wide, centered on that interval. With ratio 4 and
    window_ratio 2 this is avg_pool1d(kernel=8,stride=4,padding=2,
    count_include_pad=False).

    Fractional boundary cells contribute in proportion to their overlap.
    Integral prefix sums implement exact piecewise-constant box integration
    without interpolating feature vectors or rounding away any teacher frames.
    Crop-edge windows are clipped and renormalized over available frames.
    """
    if target.ndim != 3 or target.shape[1] < 1 or student_frames < 1:
        raise ValueError("Expected nonempty SSL features [batch,frames,channels]")
    if window_ratio <= 0:
        raise ValueError("window_ratio must be positive")
    with torch.autocast(target.device.type, enabled=False):
        values = target.detach().float().transpose(1, 2)
        teacher_frames = values.shape[-1]
        ratio = teacher_frames / student_frames
        centers = (torch.arange(student_frames, device=target.device, dtype=torch.float32) + 0.5) * ratio
        half_width = 0.5 * window_ratio * ratio
        left = (centers - half_width).clamp(0, teacher_frames)
        right = (centers + half_width).clamp(0, teacher_frames)
        prefix = F.pad(values.cumsum(dim=-1), (1, 0))

        def integral(position):
            index = position.floor().long()
            fraction = position - index
            return (
                prefix.index_select(-1, index) + values.index_select(-1, index.clamp(max=teacher_frames - 1)) * fraction
            )

        return (integral(right) - integral(left)) / (right - left)


class SemanticTeacher(nn.Module):
    """The frozen Wav2Vec2-BERT 2.0 teacher truncated to its first 16 layers, in bf16.

    ``config`` is the teacher's ``config.json``; ``state`` its 517 tensors (the encoder checkpoint's ``teacher.*``
    group). The model is built in fp32 and loaded strictly (the bf16 -> fp32 upcast is exact), then cast to bf16:
    the weights equal ``Wav2Vec2BertModel.from_pretrained(dir, dtype=torch.bfloat16)``."""

    LAYER = 16

    def __init__(self, config: dict, state: dict[str, torch.Tensor]):
        super().__init__()
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

        self.model = Wav2Vec2BertModel(Wav2Vec2BertConfig(**config)).float()
        self.model.load_state_dict(state, strict=True)
        self.model.to(torch.bfloat16)
        self.model.encoder.layers = nn.ModuleList(list(self.model.encoder.layers[: self.LAYER]))
        self.model.config.num_hidden_layers = self.LAYER
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, features, attention_mask):
        self.model.eval()
        return self.model(
            input_features=features.to(next(self.model.parameters()).dtype),
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state


@lru_cache(maxsize=32)
def _cached_resampler(input_rate: int, output_rate: int) -> T.Resample:
    """Cache polyphase kernels per worker instead of rebuilding every sample."""
    return T.Resample(int(input_rate), int(output_rate))


class Wav2Vec2BertFeaturePreprocessor:
    """Exact public W2V-BERT 2.0 fbank/stride-stack preprocessing on CPU."""

    def __init__(self, source_sample_rate: int):
        from transformers import SeamlessM4TFeatureExtractor

        self.source_sample_rate = int(source_sample_rate)
        self.target_sample_rate = 16000
        self.extractor = SeamlessM4TFeatureExtractor(
            feature_size=80,
            sampling_rate=self.target_sample_rate,
            num_mel_bins=80,
            padding_value=1,
            stride=2,
        )
        self.resampler = (
            _cached_resampler(self.source_sample_rate, self.target_sample_rate)
            if self.source_sample_rate != self.target_sample_rate
            else None
        )

    def __call__(self, audio: torch.Tensor, info: dict) -> None:
        # W2V-BERT is mono.  The production mono recipe already requests an
        # FFmpeg downmix; mean remains a harmless identity for that one channel
        # and keeps this preprocessor valid for other callers.
        mono = audio.float().mean(dim=0)
        if self.resampler is not None:
            mono = self.resampler(mono)
        encoded = self.extractor(
            mono.numpy(),
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )
        info["w2v_bert_input_features"] = encoded.input_features[0]
        info["w2v_bert_attention_mask"] = encoded.attention_mask[0].to(torch.bool)


@torch.no_grad()
def extract_ssl(audio, teacher):
    """Waveform [batch, channels, samples] at 44.1 kHz -> the teacher's layer-16 features [batch, frames, 1024]."""
    preprocessor = Wav2Vec2BertFeaturePreprocessor(44100)
    items = []
    for example in audio.detach().float().cpu():
        info = {}
        preprocessor(example, info)
        items.append(info)
    features = torch.stack([item["w2v_bert_input_features"] for item in items]).to(audio.device)
    masks = torch.stack([item["w2v_bert_attention_mask"] for item in items]).to(audio.device)
    return teacher(features, masks)
