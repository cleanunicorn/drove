# Getting started

[Docs](./README.md) / Getting started

Install drove, serve one text model, and verify a response. This tutorial uses
a local model named `qwen` and a proxy bound to `127.0.0.1:8080`. Model downloads
and the first load depend on your connection and hardware.

## Prerequisites

- **Linux or macOS.**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** on your `PATH`.
  drove needs Python 3.14 or later; uv can install that interpreter for you.
- **`llama-server`** from [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
  on your `PATH` for text and vision models. On macOS, `brew install llama.cpp`
  provides it. On Linux, use a [release binary](https://github.com/ggml-org/llama.cpp/releases)
  or build llama.cpp for your hardware.
- Enough disk space and RAM/VRAM for your chosen model. The example downloads an
  8B model at Q4_K_M quantization; choose a smaller model if needed.

Only serving audio? Go directly to [Speech-to-text](./speech-to-text.md) or
[Text-to-speech](./text-to-speech.md) for their dependencies and first request.

## 1. Install

```bash
uv tool install --python 3.14 'drove[asr] @ git+https://github.com/cleanunicorn/drove'
drove --version
```

The version command should print `drove` and a version number. `[asr]` includes
speech-to-text dependencies; omit it for a minimal install. If your shell
cannot find `drove`, run `uv tool update-shell`, then open a new terminal.

Alternatively, install from a checkout:

```bash
git clone https://github.com/cleanunicorn/drove.git
cd drove
make install
```

`make install` bootstraps uv if needed and installs the CLI as a uv tool.
`make install DROVE_EXTRAS=` omits ASR dependencies. You do not need a global
install for [development checks](./contributing.md).

## 2. Initialize local config

```bash
drove init
```

This writes and displays `~/.config/drove/config.toml`. If it already exists,
keep your settings; `drove config` displays them. For a custom path, use
`drove --config /path/to/config.toml init` and pass the same root option to
subsequent commands, or set `DROVE_CONFIG`.

## 3. Download a model

```bash
drove models download unsloth/Qwen3-8B-GGUF:Q4_K_M --name qwen
drove models list
```

The list should contain `qwen`. `--name` sets the local name you will use in
chat and API requests; `:Q4_K_M` selects a quantization without a picker.
Without `:QUANT`, repositories with several variants open an arrow-key picker
with sizes, local download status and advisory RAM estimates. Enter selects;
Esc or Ctrl+C cancels. Scripts can use `:QUANT` or `--yes` to skip it.
For other models, use the exact name printed by `drove models list`.
See [Managing models](./models.md) for naming, downloads, and tuning.

## 4. Run the server

In one terminal:

```bash
drove serve --host 127.0.0.1
```

Leave it running. In a second terminal, check the proxy and model inventory:

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models
```

Health should report `"status":"ok"`; the model list should include an object
with `"id":"qwen"`. These checks do not load the model. A server with no loaded
models is ready to accept requests.

The example uses a loopback bind. Without `--host`, drove defaults to
`0.0.0.0`, which is reachable from other machines. See
[Running drove](./operations.md#choose-a-bind-address) before enabling LAN access.

## 5. Send your first request

In the second terminal:

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Write a haiku about inference latency."}]}'
```

The response contains generated text under `choices[0].message.content`.
The first request waits for the backend to start and load the weights. Later
requests reuse it until it goes idle or is evicted. The default idle timeout is
30 minutes. If the request fails, use [Troubleshooting](./troubleshooting.md).

For terminal chat, run `drove chat qwen`; type `/help` for chat commands.

### With the OpenAI Python SDK

Install `openai` in your client project's environment. Then run:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="drove")
response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

The SDK requires a non-empty key. drove does not authenticate it. Set your
client timeout long enough for a cold model load.

## Next steps

Stop the foreground server with Ctrl+C when you are finished.

- [Text and vision](./text-and-vision.md) — streaming, images, sessions, coding agents.
- [Managing models](./models.md) — quantizations, local names, per-model settings.
- [Running drove](./operations.md) — service setup, status, request logs.
- [Configuration](./configuration.md) — timeouts, memory budget, prompt cache.

[Docs](./README.md) · [Next: Managing models →](./models.md)
