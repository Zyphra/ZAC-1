# ZAC-1

ZAC-1 (Zyphra Audio Codec 1) is the 44.1 kHz speech codec of [ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini),
as a PyTorch package (`torch.hub` or pip) and a native C++ port on ggml. It codes speech as 15 Hz frames of 10
codebooks of 4096 entries (a hop of 2940 samples). Codebook 0 is semantic: it quantizes the layer-16 features of a
frozen Wav2Vec2-BERT 2.0 teacher. Codebooks 1-9 are an acoustic residual RVQ. Decoding is causal: transformer stages
take the codes from 15 Hz to 240 Hz features, then a multi-resolution iSTFT head synthesizes the waveform. See the
[architecture section](https://www.zyphra.com/our-work/zonos3-mini#training-and-architecture-details) of the
ZONOS3-Mini post.

## Usage

```python
import soundfile as sf
import torch

decoder = torch.hub.load("Zyphra/ZAC-1", "zac_decoder", trust_repo=True)  # CUDA when available
codes = torch.randint(0, 4096, (150, 10))                                 # [frames, 10]: 10 s
audio = decoder.decode(codes) * decoder.output_gain()                     # float32 [frames * 2940] at 44.1 kHz
sf.write("decoded.wav", audio.numpy(), 44100)

encoder = torch.hub.load("Zyphra/ZAC-1", "zac_encoder", trust_repo=True)  # CUDA only
wav, sr = sf.read("speech.wav", dtype="float32", always_2d=True)
codes = encoder.encode(torch.from_numpy(wav.T), sr)                        # int32 [ceil(N / 2940), 10]
```

Or install the package:

```
pip install git+https://github.com/Zyphra/ZAC-1.git
```

```python
from zac import load_decoder, load_encoder

decoder = load_decoder("keep2_plain", device="cpu")
```

Requirements: `torch`, `torchaudio`, `transformers`, `safetensors`, `huggingface_hub` and `numpy`. pip installs them;
with `torch.hub`, install them yourself. The first load downloads the checkpoint from the Hugging Face repo
[`Zyphra/ZAC-1`](https://huggingface.co/Zyphra/ZAC-1): 0.58 GB for the default decoder, 4.07 GB for `full_mr` and
4.81 GB for the encoder. `checkpoint="path.safetensors"` (in `load_decoder`, `load_encoder` or `torch.hub.load`)
loads a local file instead.

- `decode` takes integer codes `[frames, 10]` or `[batch, frames, 10]` in `[0, 4096)` and returns float32 CPU
  audio `[frames * 2940]` or `[batch, frames * 2940]` at the raw decoder level. Multiply by `output_gain()` (3 dB
  of headroom) for playback level. It runs fp32 with TF32 off; on CUDA the transformer stages run under bf16
  autocast.
- `encode` takes a waveform `[samples]` or `[channels, samples]` at any sample rate. `prepare_audio` (also exported)
  averages the channels, resamples to 44.1 kHz with torchaudio, divides by the peak only when it exceeds 1, and keeps
  the whole 2940-sample hops within `max_seconds` (default and maximum 600). Clips shorter than 0.5 s are rejected.
  Encoding needs CUDA: the acoustic encoder runs as a CUDA graph in bf16 autocast, the teacher in bf16, and both
  quantizers in fp32. Long inputs are encoded in independent 20 s lanes with 10 s of real left context. On CPU, use
  the ggml port's `zac-encoder-cli`.

### Decoders

Two decoders share the codec, its codebooks and frame rate:

| Variant | Stage layers | Use |
| --- | --- | --- |
| `keep2_plain` (default) | decoder.0 keeps layers 0+31; stages 2/4/6/8 keep 0+11 | about 3× faster decode |
| `full_mr` | all layers | full-size |

Both were trained for 50,000 steps on top of the 600,000-step codec, with every decoder stage unfrozen.

### Streaming

```python
stream = decoder.stream()
for chunk in chunks:                  # each [n, 10] ordinary codes, in order
    play(stream.decode(chunk) * decoder.output_gain())
play(stream.flush() * decoder.output_gain())
```

- `ISTFTStream.decode(codes)` takes the next code frames and returns the samples that are now final. After F frames,
  exactly `2940F − 1470` samples have been emitted; `flush()` brings the total to `2940F`. The concatenated output
  equals `decode` of the whole sequence to float32 rounding.
- A residual code (codebook 1-9) of `-1` means "missing": that codebook's contribution is left out of the frame,
  giving a coarser RVQ decode (the codec was trained with quantizer dropout).
- `fork()` returns an independent stream that continues from the current state.
- `decoder.capture_stream_graphs([1, 2, 4, 8])` CUDA-graphs the chunk shapes of a stream that decodes those chunk
  sizes from the start. Later `stream()` calls replay them bit-identically to the eager decode; other shapes run
  eagerly. It is a no-op off CUDA.

## Checkpoint format

Three safetensors files (format `zac-1/1`). Every file carries `format`, `kind` (`encoder` or `decoder`) and the JSON
configs its modules are built from (`model_config`, plus `teacher_config` or `head_config`) in its metadata.

| File | Tensors |
| --- | --- |
| `zac-1-encoder.safetensors` | `codec.*`: acoustic encoder, semantic encoder and head, quantizers, 44100→44160 Hz resampler (fp32); `teacher.*`: the 517 Wav2Vec2-BERT 2.0 tensors (bf16) |
| `zac-1-decoder-keep2_plain.safetensors` | `codec.*`: quantizers, semantic head, decoder stages 0-8, iSTFT head, 44160→44100 Hz resampler (fp32) |
| `zac-1-decoder-full_mr.safetensors` | the same for `full_mr` |

`scripts/build_checkpoint.py` builds them from the ZONOS3-Mini release weights (`autoencoder/` of
[`Zyphra/ZONOS3-Mini`](https://huggingface.co/Zyphra/ZONOS3-Mini)) and optionally uploads them:

```
uv run python scripts/build_checkpoint.py --generator autoencoder/generator.pt --heads autoencoder/heads \
  --teacher autoencoder/teacher --out-dir dist [--push Zyphra/ZAC-1]
```

## Native ggml version

`ggml/` is a C++17 port of the decoder, the encoder and the semantic teacher on
[ggml](https://github.com/ggml-org/ggml), with the submodule `ggml/ggml` pinned to v0.27.0 (`a05c9922`). It runs on
CPU, Metal or CUDA, without Python.

### Build and models

```
git submodule update --init ggml/ggml
cmake -B ggml/build -S ggml && cmake --build ggml/build -j                           # CPU (and Metal on macOS)
cmake -B ggml/build-cuda -S ggml -DGGML_CUDA=ON && cmake --build ggml/build-cuda -j  # CUDA
hf download Zyphra/ZAC-1 --include 'gguf/*' --local-dir models
```

CUDA builds target sm_90 (H100) unless `-DCMAKE_CUDA_ARCHITECTURES=...` says otherwise. Configure applies the
patches in `ggml/patches` to the submodule's worktree (see [ggml/patches/README.md](ggml/patches/README.md)); re-run
configure after resetting the submodule.

| GGUF (in `Zyphra/ZAC-1`, `gguf/`) | Size | Weights |
| --- | ---: | --- |
| `zac-1-decoder-keep2_plain-f16.gguf` | 0.45 GB | `keep2_plain`: transformer, semantic-head and iSTFT-head matrices F16, the rest F32 |
| `zac-1-decoder-full_mr-f16.gguf` | 2.20 GB | `full_mr`, same storage |
| `zac-1-encoder-f32.gguf` | 4.03 GB | encoder, F32 |
| `zac-1-ssl-teacher-f32.gguf` | 1.55 GB | Wav2Vec2-BERT 2.0 teacher (16 layers), F32 |

Their tensors and keys are byte-identical to the ZONOS3-Mini GGUFs (`gguf/zac-keep2.gguf`, `gguf/zac.gguf`,
`gguf/zac-encoder.gguf`, `gguf/ssl-teacher.gguf` in `Zyphra/ZONOS3-Mini`), apart from `general.name`, so either set
works with either runtime.

`scripts/convert_gguf.py {decoder --variant V,encoder,ssl-teacher} [--checkpoint F] [--outtype T] [-o OUT]` writes
them from the safetensors checkpoints (`uv run python scripts/convert_gguf.py ...`; `gguf` is in the dev group).

### CLI

```
ggml/build/zac-cli models/gguf/zac-1-decoder-keep2_plain-f16.gguf codes.npy decoded.wav [--gpu|--cpu]
ggml/build/zac-encoder-cli models/gguf/zac-1-encoder-f32.gguf models/gguf/zac-1-ssl-teacher-f32.gguf \
  --audio speech.mp3 codes.npy [--gpu|--cpu]
ggml/build/ssl-encoder-cli models/gguf/zac-1-ssl-teacher-f32.gguf --audio speech.mp3 --out dir [--cpu]
```

- `zac-cli <decoder.gguf> <codes.npy> <out.wav|out.npy> [--sheared [--eos N] [--skip-frames N]] [--headroom-db X]
  [--gpu|--cpu]` decodes ordinary codes `[frames, 10]` (`<i4`, `<i8` or `<f4`) to a 16-bit WAV or a float32 `.npy`
  at playback level. `--headroom-db 0` gives the raw decoder level. `--sheared` reads sheared stream codes instead
  (codebook c of frame t in row t+c, `<f4`), with an optional `<codes>.eos.npy`.
  `zac-cli --seam <decoder.gguf> <sheared-codes.npy> [--block N]` checks streamed against whole decoding.
- `zac-encoder-cli <encoder.gguf> <ssl-teacher.gguf> --audio <file|pcm.npy> <out.npy> [--gpu|--cpu] [--dump DIR]`
  writes ordinary `<i4` codes `[frames, 10]`, which `zac-cli` decodes directly. The audio is any file ffmpeg reads
  (decoded to 44.1 kHz mono) or a float32 44.1 kHz `.npy`, up to 600 s. Unlike `prepare_audio`, the input is not
  peak-normalized.
- `ssl-encoder-cli` dumps the teacher's frontend features and layer outputs for comparison.
- `ZAC_THREADS` sets the CPU thread count. `ZAC_FFMPEG` names the ffmpeg binary; otherwise one next to the
  executable is used, then `ffmpeg` on `PATH`.

The C++ API is `ggml/src/zac.h` (`zac_load`, `zac_decode`, `zac_stream_decode`), `ggml/src/zac-encoder.h` and
`ggml/src/ssl-encoder.h`, linked from the static library `zac`.

### Accuracy

Measured on one H100 node (`ggml/build-cuda`, `--cpu` and `--gpu` backends) against the Python package on the same
inputs.

Decoder (`decoder_parity.py`, 150 seeded random frames, f16 GGUFs, against the Python decoder on CPU fp32, raw level):

| Variant | Backend | cos | max abs |
| --- | --- | ---: | ---: |
| `keep2_plain` | CPU | 0.99999827 | 0.0094 |
| `keep2_plain` | CUDA | 0.99999847 | 0.0071 |
| `full_mr` | CPU | 0.99999928 | 0.0032 |
| `full_mr` | CUDA | 0.99999908 | 0.0045 |

Encoder (`encoder_parity.py`, ZONOS3-Mini's `AmericanFemale.mp3` default voice tiled to 66.1 s, f32 GGUFs, against
the Python encoder on CUDA). Code agreement:

| Backend | < 60 s | codebook 0, < 60 s | > 60 s | codebook 0, > 60 s |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.803 | 0.968 | 0.750 | 0.957 |
| CUDA | 0.829 | 0.979 | 0.848 | 0.989 |

- **The decoder is not bit-identical**, but within float rounding of Python. The f16 GGUF matrices account for most
  of the difference.
- **Encoder codes are not bit-exact.** BF16 rounding inside cuDNN and fused attention flips RVQ choices at codeword
  boundaries, so the deeper residual codebooks agree less (per codebook on CUDA: 0.98, 0.93, 0.89, 0.86, 0.85, 0.80,
  0.76, 0.75, 0.75, 0.74). The Python package on CUDA is the reference; it is deterministic and reproduces ZONOS3's
  encoder exactly.

### ggml tests

```
ctest --test-dir ggml/build          # zac-seam, zac-stream (+ ssl-reference-state with -DZAC_TEST_SSL_TEACHER=<ssl-teacher.gguf>)
uv run python ggml/tests/parity/decoder_parity.py --build ggml/build --gguf models/gguf/zac-1-decoder-keep2_plain-f16.gguf
uv run python ggml/tests/parity/encoder_parity.py --build ggml/build-cuda --gpu \
  --encoder-gguf models/gguf/zac-1-encoder-f32.gguf --teacher-gguf models/gguf/zac-1-ssl-teacher-f32.gguf --audio speech.mp3
```

`decoder_parity.py` decodes seeded random codes (or `--codes`) with the Python decoder on CPU and with `zac-cli`, and
fails below cosine 0.999. `encoder_parity.py` (needs CUDA) tiles a clip to `--seconds` (default 66.1) and fails below
0.65 code agreement overall or 0.9 on codebook 0, before or after 60 s. `ggml/scripts/make-dummy-zac.py` regenerates
the test fixtures.

## Tests

```
uv run pytest -q
```

The tests need no weights. The development environment (`uv sync`) is locked to Python 3.12, torch and torchaudio
2.9.1 (PyPI's cu128 build), transformers 4.57.3, safetensors 0.7.0 and huggingface-hub 0.36.0, through `[tool.uv]
constraint-dependencies`. Those constraints do not apply to consumers.

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Third-party components:

| Component | Where | License |
| --- | --- | --- |
| Wav2Vec2-BERT 2.0 weights ([facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)) | the first 16 layers: `teacher.*` in the encoder checkpoint, the ssl-teacher GGUF | MIT |
| [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) | `zac/layers.py` (adapted) | MIT |
| [EnCodec](https://github.com/facebookresearch/encodec) padding helpers | `zac/s1.py` | MIT |
| MOSS-Audio-Tokenizer-v2 (OpenMOSS-Team) | architecture reference for `zac/moss.py` | Apache-2.0 |
| [ggml](https://github.com/ggml-org/ggml) | `ggml/ggml` submodule | MIT |
| [pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft) | `ggml/vendor/pocketfft_hdronly.hpp` | BSD-3-Clause |

The MIT license texts are in [licenses/](licenses).

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
