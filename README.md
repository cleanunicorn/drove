# drove

**drove — local LLMs on demand.**

A llama.cpp server manager that wakes models when you need them and shuts them down when you don't. It proxies an OpenAI-compatible API, lazily starts `llama-server`, and keeps configuration transparent.

## Install

### Install script

Install `drove` with the repository install script. It installs [`uv`](https://docs.astral.sh/uv/) if needed, then installs the `drove` CLI as a `uv` tool.

```bash
curl -fsSL https://raw.githubusercontent.com/cleanunicorn/drove/main/install.sh | bash
```

Or run the same script from a local checkout:

```bash
git clone https://github.com/cleanunicorn/drove.git
cd drove
./install.sh
```

After installation, make sure the `uv` tool bin directory is on your `PATH` if the installer prints a PATH warning. `drove` also requires `llama-server` from llama.cpp before you start the proxy.

## Quick start

```bash
drove init
drove models download unsloth/Qwen3-8B-GGUF
drove serve &
drove chat
```

## Flagship features

- **Lazy by design** — model processes start on first request and stop after idle timeout.
- **OpenAI-compatible** — drop drove behind existing OpenAI SDK clients.
- **Observable** — request logging plus TUI/web inspection for request/response debugging.
- **Speech-to-text** — serve ASR models like NVIDIA Parakeet via `/v1/audio/transcriptions` with the built-in ONNX worker ([docs](./docs/speech-to-text.md)).

## Comparison

| | drove | Ollama | llama.cpp directly |
|--|--|--|--|
| Backend | llama.cpp | llama.cpp (forked) | llama.cpp |
| Lazy model loading | yes | yes | no |
| Multiple concurrent models | yes | yes | manual |
| OpenAI-compatible API | yes | yes | yes (server) |
| Direct llama-server flags | yes (per model) | partial | yes |
| HuggingFace download + GGUF convert | yes | partial | manual |
| Request/response observability | built-in | no | no |
| TUI chat with sessions | yes | no | no |
| Configuration surface | TOML + env | env + Modelfile | flags |

## Documentation

- In-repo docs: [`docs/`](./docs/README.md)
- Hosted docs target: `https://drove.dev/docs`

## Reporting issues

When opening a new issue, please use the repository issue templates:

- [Bug report](./.github/ISSUE_TEMPLATE/bug_report.md)
- [Feature request](./.github/ISSUE_TEMPLATE/feature_request.md)

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src/
```
