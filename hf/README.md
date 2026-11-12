---
license: apache-2.0
library_name: pytorch
tags:
- audio
- audio-codec
- speech
- neural-codec
- gguf
- zonos3
---

# ZAC-1 (Zyphra Audio Codec 1)

The 44.1 kHz speech codec of [ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini): 15 Hz frames of 10 codebooks
of 4096 entries. Codebook 0 is semantic (distilled from a frozen Wav2Vec2-BERT 2.0 teacher), codebooks 1-9 are an
acoustic residual RVQ, and a causal multi-resolution iSTFT decoder synthesizes the waveform. Under similar data
constraints, a TTS model trained with ZAC-1 achieves a 10x reduction in word error rate compared with one trained with
DAC, while achieving better acoustic similarity. Compared with S1-DAC and MOSS v2, it achieves comparable results at
better compression ([ZONOS3-Mini post](https://www.zyphra.com/our-work/zonos3-mini#training-and-architecture-details)).

Code, usage, streaming and the C++/ggml CLI: [github.com/Zyphra/ZAC-1](https://github.com/Zyphra/ZAC-1).

## Files

| File | Size | Use |
| --- | ---: | --- |
| `zac-1-decoder-keep2_plain.safetensors` | 0.58 GB | PyTorch decoder, default: pruned stages (decoder.0 keeps 2 of 32 layers, stages 2/4/6/8 keep 2 of 12), about 3× faster |
| `zac-1-decoder-full_mr.safetensors` | 4.07 GB | PyTorch decoder, full-size |
| `zac-1-encoder.safetensors` | 4.81 GB | PyTorch encoder (CUDA): acoustic encoder, semantic branch, quantizers (fp32) and the Wav2Vec2-BERT 2.0 teacher (bf16) |
| `gguf/zac-1-decoder-keep2_plain-f16.gguf` | 0.45 GB | ggml decoder (`zac-cli`), `keep2_plain` |
| `gguf/zac-1-decoder-full_mr-f16.gguf` | 2.20 GB | ggml decoder (`zac-cli`), `full_mr` |
| `gguf/zac-1-encoder-f32.gguf` | 4.03 GB | ggml encoder (`zac-encoder-cli`) |
| `gguf/zac-1-ssl-teacher-f32.gguf` | 1.55 GB | ggml semantic teacher (`zac-encoder-cli`) |

sha256 of the safetensors files: `zac-1-encoder.safetensors` `89716a1f928a4de051577e664d316508ab7eac11decb5510d1bf284801fe49ee`,
`zac-1-decoder-keep2_plain.safetensors` `851ee0748dbcbf0b257944d8380c5c412418fea1628c8b2e142ae600c3c7ef40`,
`zac-1-decoder-full_mr.safetensors` `91a6d98e94e7f347606decb8444082ceec0f1fc2d971c1cef25893be6a949583`.

The weights are those of ZONOS3-Mini's codec (`autoencoder/` of `Zyphra/ZONOS3-Mini`: codec step 600,000, decoder
heads step 50,000), repackaged; the GGUFs' tensors are byte-identical to ZONOS3-Mini's.

## Usage

```python
import soundfile as sf
import torch

decoder = torch.hub.load("Zyphra/ZAC-1", "zac_decoder", trust_repo=True)  # or variant="full_mr"
audio = decoder.decode(codes) * decoder.output_gain()                     # codes [frames, 10] -> [frames * 2940]
sf.write("decoded.wav", audio.numpy(), 44100)

encoder = torch.hub.load("Zyphra/ZAC-1", "zac_encoder", trust_repo=True)  # CUDA only
wav, sr = sf.read("speech.wav", dtype="float32", always_2d=True)
codes = encoder.encode(torch.from_numpy(wav.T), sr)                        # int32 [ceil(N / 2940), 10]
```

## License

Apache-2.0. `zac-1-encoder.safetensors` (`teacher.*`) and `gguf/zac-1-ssl-teacher-f32.gguf` contain the first 16
layers of [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) (MIT).

## Citation

```bibtex
@misc{zyphra2026zonos3mini,
  title        = {{ZONOS3-Mini Technical Report}},
  author       = {Gabriel Clark and Sofian Mejjoute and George Close and Beren Millidge},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://www.zyphra.com/our-work/zonos3-mini}},
}
```
