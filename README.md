# drove

> drove was previously known as **vllama**.

**drove — local LLMs on demand.**

A llama.cpp server manager that wakes models when you need them and shuts them down when you don't. It proxies an OpenAI-compatible API, lazily starts `llama-server`, and keeps configuration transparent.

## Install

```bash
curl -LsSf drove.sh | sh
```

Or with uv:

```bash
uv tool install git+https://github.com/cleanunicorn/drove
```

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

- Docs: `drove.dev/docs` (planned)
- CLI reference currently lives in `--help` output and inline command help.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src/
```
