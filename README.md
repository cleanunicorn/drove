<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-dark.svg">
  <img alt="drove" src="./assets/logo.svg" width="360">
</picture>

### Local models on demand.

**drove wakes your models when a request comes in, and puts them back to sleep when they go idle.**
One OpenAI-compatible endpoint in front of `llama.cpp` (text + vision), a built-in ONNX worker (speech-to-text), and local text-to-speech engines.

<br>

![Python](https://img.shields.io/badge/python-3.14+-3776AB?logo=python&logoColor=white)
![OpenAI-compatible](https://img.shields.io/badge/API-OpenAI--compatible-412991?logo=openai&logoColor=white)
![Backends](https://img.shields.io/badge/backends-llama.cpp%20%2B%20ONNX%20%2B%20TTS-C45A3A)

[Quickstart](#quickstart) · [Documentation](./docs/README.md) · [Configuration](./docs/configuration.md) · [CLI](./docs/cli.md)

</div>

## Why drove?

Point an OpenAI client at `http://127.0.0.1:8080/v1`. drove starts the requested
model, forwards the request to its backend, and stops the process after it goes
idle. Count and memory budgets let you trade off keeping models ready against
freeing resources.

| Workload | Backend | Guide |
| --- | --- | --- |
| Text and vision | `llama-server` from llama.cpp | [Text and vision](./docs/text-and-vision.md) |
| Speech-to-text | Built-in ONNX worker | [Transcribe audio](./docs/speech-to-text.md) |
| Text-to-speech | Kokoro, Qwen3-TTS, Chatterbox, Fish, KittenTTS 2 | [Generate speech](./docs/text-to-speech.md) |

## Quickstart

You need [uv](https://docs.astral.sh/uv/) and, for this text example,
[`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
on your `PATH`. drove requires Python 3.14 or later; uv can download it.
Downloads and first model loads can take several minutes.

```bash
uv tool install --python 3.14 'drove[asr] @ git+https://github.com/cleanunicorn/drove'
drove init
drove models download unsloth/Qwen3-8B-GGUF:Q4_K_M --name qwen
```

Start the proxy in one terminal:

```bash
drove serve --host 127.0.0.1
```

In another terminal, send a request using the local name you chose:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Write a haiku about lazy servers."}]}'
```

Or run `drove chat qwen` for terminal chat. The first request loads the model;
it unloads after 30 minutes idle by default. Stop the foreground server with
Ctrl+C. The example binds to loopback; the configured default is `0.0.0.0`
(all interfaces). See [Running drove](./docs/operations.md) for network and
service setup.

For installation alternatives, SDK setup, and checks at each step, follow
[Getting started](./docs/getting-started.md).

## What you can do

- **Manage resources:** load models on demand, evict least-recently-used models,
  and persist llama.cpp prompt caches across idle shutdowns.
- **Bring your own models:** download from Hugging Face, pick quantizations,
  resume downloads, and set llama-server arguments per model.
- **Use familiar clients:** connect an OpenAI SDK, the built-in chat TUI, or
  pi, opencode, and Kilo Code via `drove integrate`.
- **Inspect requests:** watch server status, browse optional request logs in
  the terminal or web UI, and apply supported config changes live.

```text
OpenAI client / curl / drove chat
              │
              ▼
      drove proxy :8080
              │  start the requested model if needed
              ▼
   llama-server / ONNX worker / TTS worker
              │
              ▼
     response → idle timer → shutdown
```

See [Architecture](./docs/architecture.md) for lifecycle and backend details.

## Documentation

Start with the [documentation hub](./docs/README.md), or pick a task:

| I want to… | Read |
| --- | --- |
| Install drove and make my first request | [Getting started](./docs/getting-started.md) |
| Download, name, inspect, or tune a model | [Managing models](./docs/models.md) |
| Send text, images, or connect a coding agent | [Text and vision](./docs/text-and-vision.md) |
| Run a service or inspect traffic | [Running drove](./docs/operations.md) |
| Look up commands, settings, or endpoints | [CLI](./docs/cli.md) · [Configuration](./docs/configuration.md) · [HTTP API](./docs/api.md) |
| Diagnose a failed request | [Troubleshooting](./docs/troubleshooting.md) |

The Markdown files in this repository are the canonical docs. The hosted
`drove.dev/docs` site is planned.

## Development

From a checkout with uv installed:

```bash
make lint
make typecheck
make test
```

The targets sync the development environment automatically. See
[Contributing](./docs/contributing.md) for the branch and PR workflow, and
[Release process](./docs/deploy.md) for publishing.

[Report a bug](./.github/ISSUE_TEMPLATE/bug_report.md) ·
[Request a feature](./.github/ISSUE_TEMPLATE/feature_request.md)
