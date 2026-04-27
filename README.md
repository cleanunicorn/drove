# drove

> **drove was previously known as `vllama`.** Same project, new name. See the [migration note](#migration-from-vllama).

Local LLMs on demand. A `llama.cpp` server manager that wakes models when you need them and shuts them down when you don't.

drove starts your local language models the way a process manager should: lazily, transparently, and without making decisions for you. It pulls models from HuggingFace, starts a `llama-server` process on first request, proxies an OpenAI-compatible API, and shuts the model down after it goes idle.

It's for people who would rather pass `--n-gpu-layers` themselves than have it guessed. If Ollama is the friendly all-in-one, drove is the lever-exposed proxy that sits one layer closer to llama.cpp.

## Install

```bash
curl -LsSf https://raw.githubusercontent.com/cleanunicorn/drove/master/install.sh | sh
```

Requires [`llama.cpp`](https://github.com/ggml-org/llama.cpp/) on your `PATH` (`llama-server --version` should work).

## Quick start

```bash
drove init
drove models download unsloth/Qwen3-8B-GGUF
drove server &
drove chat
```

## Features

- **Lazy by design.** Models start on first request, sleep after idle.
- **Multiple concurrent models.** Each on its own `llama-server` instance.
- **OpenAI-compatible API.** Drop drove in front of any OpenAI client.
- **HuggingFace integration.** `drove models download <repo>` — safetensors auto-convert to GGUF.
- **Per-model config.** Context size, GPU layers, every llama.cpp flag.
- **Observability.** Log every request, browse in TUI or web UI.
- **Interactive chat.** TUI with session persistence and tool calling.

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

Full reference, configuration, and guides: see [`docs/`](docs/) or run `drove --help`.

## Development

```bash
git clone https://github.com/cleanunicorn/drove
cd drove
uv sync
uv run drove --help
```

```bash
make test       # pytest
make lint       # ruff
make typecheck  # mypy
make fmt        # ruff format
```

## Migration from vllama

If you previously installed `vllama`, the first run of `drove` automatically migrates:

- `~/.config/vllama/` → `~/.config/drove/`
- `~/.local/share/vllama/` → `~/.local/share/drove/`

Environment variables are renamed `VLLAMA_*` → `DROVE_*`. Update any scripts or systemd units that set them.

The old `vllama` binary can be uninstalled with `uv tool uninstall vllama`.

## License

See [LICENSE](LICENSE).
