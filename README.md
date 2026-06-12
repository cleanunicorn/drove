# drove

**drove — local models on demand.**

A local model server manager that wakes models when you need them and shuts them down when you don't. It proxies an OpenAI-compatible API and lazily starts the right backend per model: `llama-server` for text generation (GGUF), or the built-in ONNX worker for speech-to-text (e.g. NVIDIA Parakeet). Configuration stays transparent.

## Install

Install `drove` from a checkout with `make install`. It installs [`uv`](https://docs.astral.sh/uv/) if needed, then installs the `drove` CLI (with speech-to-text support) as a `uv` tool:

```bash
git clone https://github.com/cleanunicorn/drove.git
cd drove
make install
```

Or install directly with `uv` without cloning:

```bash
uv tool install 'drove[asr] @ git+https://github.com/cleanunicorn/drove'
```

After installation, make sure the `uv` tool bin directory is on your `PATH` if `make install` prints a PATH warning. `drove` also requires `llama-server` from llama.cpp before you start the proxy.

## Quick start

```bash
drove init
drove models download unsloth/Qwen3-8B-GGUF
drove serve &
drove chat
```

## Text generation

Download any GGUF model from HuggingFace and chat with it through the TUI or the OpenAI-compatible API:

```bash
drove models download unsloth/gemma-3-12b-it-GGUF:Q4_K_M
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/gemma-3-12b-it-GGUF:Q4_K_M",
    "messages": [{"role": "user", "content": "Write a haiku about lazy servers."}]
  }'
```

The model loads on the first request and shuts down after the idle timeout. Any OpenAI SDK client works — point it at `http://localhost:8080/v1`.

## Speech-to-text

drove also serves ASR models such as NVIDIA Parakeet through the same port and lifecycle, using its built-in ONNX worker (no extra server binary). Speech-to-text support is included by `make install`; if you installed drove manually, add the `asr` extra (`pip install 'drove[asr]'`). Download an ONNX export:

```bash
drove models download istupakov/parakeet-tdt-0.6b-v3-onnx
```

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F model='istupakov/parakeet-tdt-0.6b-v3-onnx' \
  -F file=@speech.wav
```

```json
{"text": "And so, my fellow Americans, ask not what your country can do for you ..."}
```

Text and speech models are managed identically (`drove models list/info/config/delete`) and can be loaded side by side. See the [speech-to-text docs](./docs/speech-to-text.md) for model configuration, supported formats, and OpenAI SDK usage.

## Flagship features

- **Lazy by design** — model processes start on first request and stop after idle timeout.
- **OpenAI-compatible** — drop drove behind existing OpenAI SDK clients.
- **Observable** — request logging plus TUI/web inspection for request/response debugging.
- **Speech-to-text** — serve ASR models like NVIDIA Parakeet via `/v1/audio/transcriptions` with the built-in ONNX worker ([docs](./docs/speech-to-text.md)).

## Comparison

| | drove | Ollama | llama.cpp directly |
|--|--|--|--|
| Backend | llama.cpp + ONNX (ASR) | llama.cpp (forked) | llama.cpp |
| Lazy model loading | yes | yes | no |
| Multiple concurrent models | yes | yes | manual |
| OpenAI-compatible API | yes | yes | yes (server) |
| Speech-to-text models | yes (built-in worker) | no | no |
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
