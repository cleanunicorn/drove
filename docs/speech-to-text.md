# Speech-to-text

[Docs](./README.md) / Speech-to-text

drove can serve speech-to-text (ASR) models — such as NVIDIA Parakeet TDT —
alongside LLMs, with the same lazy lifecycle: the model loads on the first
request and shuts down after the idle timeout.

Instead of `llama-server`, ASR models run in drove's built-in worker
(`python -m drove.workers.asr`), which loads ONNX models through
[onnx-asr](https://github.com/istupakov/onnx-asr) and exposes an
OpenAI-compatible `POST /v1/audio/transcriptions` endpoint. Requests go through
the same drove port as chat completions; the `model` form field picks the model.

## Install

Clone the repository and install drove with speech-to-text support:

```bash
git clone https://github.com/cleanunicorn/drove.git
cd drove
make install
drove init
```

`make install` bootstraps uv if needed and includes ASR by default. If drove is
already installed without ASR, rerun `make install` from your checkout.
No `llama-server` binary is needed for transcription.

`ffmpeg` is recommended (but not required) so the worker can accept compressed
audio formats (mp3, m4a, ogg, …). Without it, uploads must be WAV.

## Download a model

Use ONNX exports of ASR models, e.g. the Parakeet ONNX repos:

```bash
drove models download istupakov/parakeet-tdt-0.6b-v3-onnx --name parakeet
# smaller int8 variant:
drove models download istupakov/parakeet-tdt-0.6b-v3-onnx:int8 --name parakeet-int8
```

For known repos, drove auto-configures the ASR model type at download time and
stores it in the sidecar config (`asr_model`). For other repos, set it
manually:

```bash
drove models config 'my-asr-model' asr_model nemo-parakeet-tdt-0.6b-v3
```

## Transcribe

Start the proxy in one terminal with `drove serve --host 127.0.0.1`. In another
terminal, call the audio API with the local name from `drove models list`:

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=parakeet \
  -F file=@speech.wav
```

Or install `openai` in your client environment and use the SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
with open("speech.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="parakeet", file=f
    )
print(result.text)
```

Supported `response_format` values: `json` (default), `text`, `verbose_json`.
Uploads are limited to 100 MB; larger uploads return HTTP 413.

## How backend selection works

drove picks the backend per model:

- `.gguf` primary file → `llama-server`
- `.onnx` primary file → built-in ASR worker
- explicit override: `drove models config <name> backend asr`

ASR-specific per-model config keys:

| Key | Meaning |
|--|--|
| `backend` | Force a backend (`llama`, `asr`, or `tts`) |
| `asr_model` | onnx-asr model type, e.g. `nemo-parakeet-tdt-0.6b-v3` |
| `asr_quantization` | Quantization variant to load, e.g. `int8` |

Everything else — idle shutdown, `max_loaded_models` eviction, `/status`,
request observability — applies to ASR models exactly as it does to LLMs.

[Docs](./README.md) · [CLI](./cli.md) · [Configuration](./configuration.md) · [Troubleshooting](./troubleshooting.md)
