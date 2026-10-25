# Configuration

`drove` reads global settings from:

- `~/.config/drove/config.toml`
- or a custom path set with `DROVE_CONFIG`

Environment variables with the `DROVE_` prefix override file values.

## Example config

```toml
models_dir = "~/.local/share/drove/models"
listen_host = "0.0.0.0"
listen_port = 8080
llama_server_bin = "llama-server"
speech_timeout_seconds = 600
idle_timeout_seconds = 1800
max_loaded_models = 1
max_memory = "24GB"
prompt_cache = true
prompt_cache_ttl_seconds = 3600

[llama_server]
n_gpu_layers = -1
```

## Default model

Set `default_model` to a model ID shown by `drove models list`:

```toml
default_model = "org/model:Q4_K_M"
```

Or use `drove config default_model "org/model:Q4_K_M"` after `drove init`.
`DROVE_DEFAULT_MODEL` overrides the file value. The default is unset; remove
its TOML key or run `drove config default_model ""` to clear it. A blank
environment value also means unset.

An explicit API `model` or chat model argument always wins. When an API
request omits `model`, drove lazily starts the configured default, even if
another model is already loaded. If that default is missing, the request
returns an actionable 404; it does not silently select another model. With
no default, the server keeps its existing loaded-model fallback (or returns
400 when none is loaded). The request body is forwarded unchanged. This
setting applies across proxy endpoints, so speech requests should explicitly
name a suitable speech model when the default is a chat model.

`drove chat` without a model argument uses the configured default before
opening the picker. This also applies to `--endpoint`: the identifier is sent
to that endpoint without requiring a local model file. Pass an explicit model
to override it. Resuming a session uses the selected model.

Changes and removal take effect for subsequent server requests after config
reload, without restarting. Models are validated when used, allowing you to
configure a default before downloading it; startup does not preload it.

## Speech request timeout

`speech_timeout_seconds` controls how long the proxy waits for a response from
`/v1/audio/speech` (default `600`, a positive number of seconds). Speech workers
buffer the complete audio before responding, so long inputs on CPU can take
more than five minutes. Increase the read timeout for longer jobs:

```bash
drove config speech_timeout_seconds 3600
```

The setting also accepts `DROVE_SPEECH_TIMEOUT_SECONDS` and applies to new
requests after config reload. Clients need their own timeout long enough to
wait for synthesis. `startup_timeout_seconds` separately controls waiting for
the worker to become healthy. Other endpoints retain their 300-second read
timeout; connection, write, and pool timeouts also remain 300 seconds.

## Model eviction

Two independent limits control when loaded models are stopped to make room
for a newly requested one (in both cases the least-recently-used idle model
is evicted first; models with in-flight requests are drained before being
stopped):

- `max_loaded_models` — how many models may be loaded at once (`0` = unlimited, default `1`).
- `max_memory` — combined memory budget for all loaded models (`"0"` = unlimited, the default).
  Accepts decimal (`"24GB"`, `"512MB"`) and binary (`"16GiB"`) units, or a plain
  number of bytes.

The memory used by a model is estimated from its on-disk file size (all shards
for sharded GGUF models, all `.onnx` files for speech-to-text models).
Text-to-speech models reserve a fixed per-engine estimate instead — 3 GiB for
Kokoro, 4 GiB for Qwen3-TTS and Chatterbox, 8 GiB for Fish, 6 GiB for KittenTTS 2 — which the
`tts_memory_mb` model key overrides. Context
(KV cache) and runtime overhead are not counted, so leave some headroom below
your real RAM/VRAM limit. A model whose estimate alone exceeds `max_memory` is
still started (after evicting everything else) rather than refused.

## Prompt cache

llama-server keeps a prompt (KV) cache in RAM so a follow-up request only
processes the tokens that changed. That cache dies with the process — and drove
stops the process on every idle timeout, so the first request after a model
wakes up re-processes the whole prompt.

drove carries it across sleep/wake: each slot's KV cache is written to disk
before a model is stopped and restored as soon as the model is healthy again.
This is on by default — set `prompt_cache = false` to turn it off.

```toml
prompt_cache = true                  # default: true
prompt_cache_dir = "~/.local/share/drove/prompt-cache"
prompt_cache_ttl_seconds = 3600      # discard cache files older than this; 0 = never expire
prompt_cache_timeout_seconds = 60    # max wait for a single save or restore
```

Sliding-window models (Gemma and friends) only keep the attention window, so
their saved cache is unusable unless the full SWA cache is kept. drove reads the
GGUF header at startup and passes `--swa-full` only for models that actually
declare a sliding window; everything else is started unchanged. Setting
`swa_full` in a model's config overrides the decision either way.

For those models the flag is the one real cost of the prompt cache: a full-size
SWA cache needs more memory than a windowed one (measured on `gemma-4-E4B` at
32k context: 5.8 GB → 7.0 GB resident). Set `swa_full = false` on a model to
keep the smaller cache and give up the reuse. What it buys, measured on the same
model at 2048 context with a 20-token prompt:

| | first request | after idle shutdown + wake |
|---|---|---|
| prompt cache off | 20 tokens processed | 20 tokens processed |
| prompt cache on | 20 tokens processed | 1 processed, 19 from cache |

Notes:

- Cache files are large — roughly the KV size of the cached prompt (tens of MB
  for a short conversation, hundreds for a long one), one file per busy slot per
  model. The TTL is what keeps the directory bounded; expired files are deleted
  when drove starts and before each model start.
- Restoring only helps when the next prompt shares a prefix with the cached one.
- A cache file the server refuses — because `ctx_size`, `cache_type_k/v` or the
  slot count changed since it was written — is deleted rather than retried.
- The cache is an optimization: every failure to save or restore is logged and
  otherwise ignored, and speech-to-text and text-to-speech models skip it entirely.

## Per-model config

Each model can have a sidecar config file in the models directory:

- model file: `~/.local/share/drove/models/<name>.gguf`
- config file: `~/.local/share/drove/models/<name>.toml`

Example:

```toml
ctx_size = 4096
n_gpu_layers = -1
```

Only keys declared in `ModelConfig` are accepted; unknown keys are silently ignored.

Supported keys:

- Context and memory: `ctx_size`, `n_gpu_layers`, `main_gpu`, `tensor_split`, `load_mode`
- Batching: `batch_size`, `ubatch_size`, `n_parallel` (→ `--parallel`)
- Sampling defaults: `temp`, `top_p`, `top_k`
- Performance: `threads`, `threads_batch`, `flash_attn`
- Prompt cache: `cache_prompt`, `cache_reuse`, `cache_ram`, `ctx_checkpoints`,
  `context_shift`, `swa_full`
- KV quantization: `cache_type_k`, `cache_type_v`
- Rope scaling: `rope_freq_base`, `rope_freq_scale`
- Multimodal: `mmproj`
- Escape hatch: `extra_args`

### Speeding up prompt processing

llama-server caches prompts by default (`cache_prompt`) and keeps up to
`cache_ram` MiB of them in memory. The flag worth adding by hand is
`cache_reuse`: it lets the server reuse cached blocks even when the prompt
changed in the middle, which is the common case for agents and chat clients that
rewrite earlier turns.

```toml
cache_reuse = 256    # min chunk size to reuse via KV shifting; 0 (default) = off
cache_ram = 16384    # MiB of prompt cache to keep in RAM (-1 = no limit, 0 = off)
```

### Raw llama-server arguments

`extra_args` is passed through verbatim, after every argument drove generated —
llama-server takes the last occurrence of a flag, so these win:

```toml
extra_args = ["--no-webui", "--lora", "/models/adapter.gguf"]
```

From the CLI the value is split shell-style:

```bash
drove models config mymodel extra_args "--lora /models/adapter.gguf"
```

Drove-specific keys (never passed to llama-server): `backend` (`llama`, `asr`, or
`tts`), `asr_model`, `asr_quantization` — see [Speech-to-text](./speech-to-text.md) —
and the text-to-speech keys below.

### Text-to-speech keys

`drove models download` and `drove models add-tts` write these for you; set them
by hand or with `drove models config <name> <key> <value>` to change them.

| Key | Meaning |
|--|--|
| `tts_engine` | `kokoro`, `qwen`, `chatterbox`, `fish`, or `kitten`. |
| `tts_voice` | Default voice: a preset (`af_heart`, `Ryan`, …), an alias (`male`, `female`, `british-male`, `british-female`), a Kokoro blend (`af_heart:2,af_bella:1`), or an absolute reference WAV path. Defaults to `female`. |
| `tts_device` | `auto` (CUDA if available, else CPU), `cpu`, `cuda`, or `mps`. Defaults to `auto`. |
| `tts_model_dir` | Directory with downloaded weights, relative to the sidecar or absolute. Unset means the Hugging Face cache. |
| `tts_language` | Qwen3-TTS language, e.g. `English`. |
| `tts_max_chunk_chars` | Maximum characters synthesized per chunk (at least 150). |
| `tts_normalize` | KittenTTS 2 English text normalization. Defaults to false for multilingual text. |
| `tts_preset` | KittenTTS 2 decoding: `stable` (default) or `expressive`. |
| `tts_memory_mb` | Memory reserved against `max_memory` while the model is loaded. |

```toml
# ~/.local/share/drove/models/qwen-tts.toml
backend = "tts"
tts_engine = "qwen"
tts_model_dir = "."
tts_voice = "Ryan"
tts_language = "English"
tts_device = "cuda"
tts_memory_mb = 6144
```

These keys also work as global model defaults (`drove models config tts_device cpu`),
and per-model sidecars take precedence.

The first start of each engine installs its Python environment and may download
weights, which can take longer than the default `startup_timeout_seconds` (300).
Raise it before first use:

```bash
drove config startup_timeout_seconds 1800
```
