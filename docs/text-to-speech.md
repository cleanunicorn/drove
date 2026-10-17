# Text-to-speech

Drove integrates Cathy's local Kokoro, Qwen3-TTS, Chatterbox-Nano, and Fish
S1-mini engines behind the OpenAI speech API. Synthesis stays on your machine.
Download weights ahead of time with `drove models download`, just as for GGUF
and ONNX models. Engine dependencies are installed on first use.
Fish requires accepting its model license and authenticating Hugging Face
downloads, as in Cathy.

## Quick start

Install `uv` and `espeak-ng`, plus `ffmpeg` for compressed audio and speed
conversion. Qwen also requires the `sox` system package.

```bash
drove models download hexgrad/Kokoro-82M --name kokoro
drove serve
```

```bash
curl http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","input":"Hello from Drove.","voice":"alloy","response_format":"wav"}' \
  --output speech.wav
```

The official OpenAI client works with the same base URL used for chat and
transcription:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    input="Hello from Drove.",
    voice="alloy",
    response_format="mp3",
) as response:
    response.stream_to_file("speech.mp3")
```

Synthesis completes before audio is sent. Streaming clients can consume that
binary response; incremental synthesis and SSE events are not implemented.

## Registration and lifecycle

The regular download command recognizes these repositories automatically:

```bash
drove models download hexgrad/Kokoro-82M --name kokoro
drove models download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --name qwen-tts
drove models download ResembleAI/chatterbox-nano --name chatterbox
drove models download fishaudio/s1-mini --name fish
```

The standard confirmation, progress display, resume behavior, and `--yes`
option apply. Complete files are skipped on subsequent runs. TTS quantization
tags are not supported; use `--tts-engine kokoro|qwen|chatterbox|fish` for an
otherwise compatible repository that is not in the automatic detection table.

Downloads keep the repository's directory structure, including voice packs and
nested tokenizer files. A `model.tts` entry and a normal `model.toml` sidecar
configure the engine and `tts_model_dir = "."`. Workers load the downloaded
checkpoints and Kokoro voices directly from this directory. `models list`,
`info`, `config`, `delete`, repository aliases, and `/v1/models` work normally.
Deleting a downloaded model removes its managed files. Speech models are
excluded from chat-agent integrations.

You can also use `drove models add-tts NAME --engine ENGINE` to register an
engine whose weights download automatically into the shared Hugging Face
cache on first use. Deleting such a registration preserves the shared caches.
`--voice` and `--device` set its startup defaults. For downloaded models, use
`drove models config NAME tts_voice female` and
`drove models config NAME tts_device cpu` (or `auto`, `cuda`, `mps`). `auto`
uses CUDA when available, otherwise CPU.

Omitting `--voice` or `--device` preserves inheritance from global model settings;
without a global setting, the worker defaults to `female` and `auto`. Explicit
registration options are saved as per-model overrides.

Chatterbox and Fish's generated male/female reference clips use a separate
Kokoro model and may download those auxiliary weights on first use. Configure
a local reference WAV with `tts_voice /absolute/path/to/reference.wav` to avoid
that step. Dependency installation and auxiliary engine assets are separate
from downloading the primary model weights.

Drove starts the worker on the first request and applies its existing health
checks, concurrent request claims, idle shutdown, model-count limit, and LRU
eviction. Prompt/KV caches apply only to llama models. Memory budgeting uses
estimated reservations for loaded engines: 3 GiB for Kokoro, 4 GiB for
Qwen/Chatterbox, and 8 GiB for Fish. Set `tts_memory_mb` for your hardware;
these are estimates rather than hard allocation limits.

Cathy's PyTorch cap requires Python 3.12; Drove continues to use Python 3.14.
`uv` builds separate cached environments from the bundled engine requirement
files, retaining Cathy's PyTorch caps and optional-engine git pins. No
separate Cathy installation is needed. First-use dependency and model downloads
can exceed Drove's default 300-second startup timeout. Increase
`startup_timeout_seconds` in global configuration before first use, for example
to `1800`. Later starts reuse the caches.

## API contract

`POST /v1/audio/speech` accepts JSON:

| Field | Behavior |
| --- | --- |
| `model` | Required registration name from `/v1/models`. |
| `input` | Required nonblank text, up to 4096 characters. |
| `voice` | Required engine voice, alias, or `alloy` for the registered default. |
| `response_format` | `mp3` (default), `opus`, `aac`, `flac`, `wav`, or `pcm`. |
| `speed` | 0.25–4.0, default 1.0. Native for Kokoro; ffmpeg conversion for other engines. |
| `instructions` | Nonempty instructions return 400 because the engines do not implement them. |
| `stream_format` | `audio` (default); SSE returns 400. |

Responses are binary audio with the matching content type. PCM is mono,
signed 16-bit little-endian at 24 kHz; WAV keeps the engine sample rate.
WAV at normal speed needs no ffmpeg; Kokoro's native speed also works without
it. Invalid requests and synthesis failures return OpenAI-style JSON errors.

## Voices

Kokoro presets include `af_heart`, `af_bella`, `af_nicole`, `am_michael`,
`am_fenrir`, `bf_emma`, `bm_george`, and `bm_fable`, plus Cathy's international
presets. Aliases include `male`, `female`, `british-male`, and `british-female`.
Weighted blends work, e.g. `af_heart:2,af_bella:1`. A Kokoro worker uses the
language of its default voice. Register a separate model for another language
or prefix:

```bash
drove models add-tts kokoro-british --voice british-male
drove models add-tts kokoro-french --voice ff_siwis
```

Qwen supports `Ryan`, `Aiden`, `Eric`, `Dylan`, `Uncle_Fu`, `Vivian`, `Serena`,
`Ono_Anna`, and `Sohee`. `male` and `female` select Ryan and Vivian.
OpenAI voice names such as `alloy`, `onyx`, and `nova` are aliases for local
voices; they do not reproduce OpenAI's voices.

Chatterbox/Fish use the registered reference voice. Use a trusted absolute WAV
path with `--voice /path/to/reference.wav`, or Cathy's male/female presets,
then send `voice="alloy"`. API callers cannot select arbitrary reference files
on the server.

For Fish, a custom reference WAV also requires a UTF-8 transcript alongside it
with the same stem, such as `reference.wav` and `reference.txt`. The text must
match the words spoken in the recording; missing or empty transcripts prevent
startup. Generated preset references supply their transcript automatically.

Kokoro accepts voices available in the downloaded model's `voices/` directory
or its Hugging Face repository, including presets beyond the curated list above.
Global model settings in `_global.toml` apply to TTS, with per-model sidecars
taking precedence, including device selection and memory reservations.

Sidecar example:

```toml
backend = "tts"
tts_engine = "kokoro"
tts_model_dir = "."  # set automatically by models download
tts_voice = "female"
tts_device = "auto"
# tts_language = "English"      # Qwen language
# tts_max_chunk_chars = 400     # >= 150, with Cathy's CUDA OOM recovery
# tts_memory_mb = 3072
```

Cathy's ebook parsing, chapter/checkpoint handling, and audiobook CLI remain
separate. The integrated engine source retains its MIT license in
`src/drove/workers/tts_runtime/CATHY-LICENSE`.
