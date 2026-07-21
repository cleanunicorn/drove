# Getting started

This guide takes you from a fresh install to serving text, vision, and
speech-to-text models through one OpenAI-compatible endpoint. For the full
reference on any topic, follow the links to the other docs.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) — used to install and run drove.
- **`llama-server`** from [llama.cpp](https://github.com/ggml-org/llama.cpp) on
  your `PATH` — required for text and vision (GGUF) models. Speech-to-text needs
  no extra binary.
- `ffmpeg` (optional) — lets the speech-to-text worker accept compressed audio
  (mp3, m4a, ogg…). Without it, upload WAV.

## 1. Install

### Option 1: uv tool install (no clone)

```bash
uv tool install 'drove[asr] @ git+https://github.com/cleanunicorn/drove'
```

Drop `[asr]` for a text-generation-only install.

### Option 2: make install (from a checkout)

```bash
git clone https://github.com/cleanunicorn/drove.git
cd drove
make install
```

`make install` installs `uv` if needed, then installs the `drove` CLI (with
speech-to-text support) as a `uv` tool. Set `DROVE_EXTRAS=` for a minimal
install. If the installer prints a `PATH` warning, add the `uv` tool bin
directory to your `PATH`.

Verify the install:

```bash
drove --version
```

## 2. Initialize local config

```bash
drove init
```

This writes `~/.config/drove/config.toml` with defaults and prints it. Override
the location with `--config` or the `DROVE_CONFIG` environment variable. See
[Configuration](./configuration.md) for every setting.

## 3. Download a model

```bash
drove models download unsloth/Qwen3-8B-GGUF
```

The reference is `org/repo` or `org/repo:QUANT`. Without a quantization tag, a
repo with several variants shows a picker. Sharded models are stored in their
own subdirectory, and partial downloads resume. List what you have:

```bash
drove models list
```

The `CAPS` column flags `vision` (multimodal) and `stt` (speech-to-text)
models. See [Managing models](#managing-models) below for more.

## 4. Run the server

```bash
drove serve
```

Leave it running. The OpenAI-compatible API is now available at:

- `GET  http://127.0.0.1:8080/v1/models`
- `POST http://127.0.0.1:8080/v1/chat/completions`
- `POST http://127.0.0.1:8080/v1/audio/transcriptions`
- `GET  http://127.0.0.1:8080/status`

Models load on the first request that names them and unload after the idle
timeout. Bind a different address with `drove serve --host 0.0.0.0 --port 9000`.

## 5. Send your first request

### From the terminal chat

```bash
drove chat
```

With no model argument, drove lists the models available on the server and lets
you pick one. Type `/help` inside the chat for commands (`/sessions`, `/theme`,
…). See [Terminal chat](#terminal-chat) for connecting to remote endpoints.

### With an OpenAI SDK

Point the base URL at drove and use any non-empty API key.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="drove")

resp = client.chat.completions.create(
    model="unsloth/Qwen3-8B-GGUF",
    messages=[{"role": "user", "content": "Write a haiku about inference latency."}],
)

print(resp.choices[0].message.content)
```

### With curl

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/Qwen3-8B-GGUF",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Speech-to-text

drove serves ASR models such as NVIDIA Parakeet through its built-in ONNX
worker — same port, same lazy lifecycle. Use an ONNX export:

```bash
drove models download istupakov/parakeet-tdt-0.6b-v3-onnx
```

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model='istupakov/parakeet-tdt-0.6b-v3-onnx' \
  -F file=@speech.wav
```

Full details — model types, quantization, `response_format` — in
[Speech-to-text](./speech-to-text.md).

## Vision / multimodal

Multimodal GGUF models ship a companion `mmproj` projector. When you download
such a repo, drove pulls the projector, records it in the model's sidecar
config, and flags the model with the `vision` capability:

```bash
drove models download unsloth/gemma-3-12b-it-GGUF   # projector auto-detected
drove models list                                   # CAPS shows "vision"
```

From there it's the standard OpenAI vision request — send image content parts to
`/v1/chat/completions` and drove forwards them to `llama-server` with the
projector loaded.

## Managing models

```bash
drove models list                            # NAME · SIZE · CAPS · CONFIG
drove models list -V                         # also show download origin
drove models info <name>                     # files, size, caps, effective config
drove models download <org/repo[:QUANT]>     # pull from HuggingFace
drove models download <ref> --name my-name   # override the local name
drove models delete <name>                   # remove a model and its config
```

Per-model configuration layers on top of the global defaults (config.toml →
global model defaults → per-model sidecar, highest wins):

```bash
drove models config <name>                   # show effective config + source of each value
drove models config <name> ctx_size 8192     # set a per-model llama.cpp flag
drove models config ctx_size 16384           # set a default for all models (global)
drove models config <name> --unset ctx_size  # remove a key
```

See [Configuration](./configuration.md) for the full list of supported keys.

## Terminal chat

`drove chat` is a TUI client with saved sessions and themes. It talks to your
local drove server by default, but can point at any OpenAI-compatible API:

```bash
drove chat                                   # pick a model from the local server
drove chat unsloth/Qwen3-8B-GGUF             # chat with a specific local model
drove chat --resume                          # resume the latest saved session
drove chat -s "You are a terse assistant."   # set a system prompt
drove chat -e https://api.openai.com/v1 -k $OPENAI_API_KEY   # remote endpoint
```

## Watch server status

Check what's loaded and how it's performing:

```bash
drove server status            # one-shot snapshot
drove server status --watch    # refresh every 2s (drove server status -w 5 for 5s)
```

It reports uptime, loaded models with idle timers, process memory/CPU, request
counts, token throughput, and time-to-first-token.

## Debug requests (observability)

Enable logging to capture every request/response pair, then browse them:

```bash
drove config observe true      # enable logging (or set observe = true in the config)
drove observe                  # interactive TUI browser (-m <model> to filter)
drove observe web              # web UI at http://127.0.0.1:8877
```

## Next steps

- [Configuration](./configuration.md) — global settings, memory budget, eviction, per-model config.
- [CLI reference](./cli.md) — every command and flag.
- [Speech-to-text](./speech-to-text.md) — ASR model types, formats, SDK usage.
- [Architecture](./architecture.md) — how the proxy and backends fit together.
