# Architecture

## Request flow

```text
Client → FastAPI proxy (drove port)
           → ServerManager.ensure_running(model)
               → if not running: spawn backend subprocess
                   (GGUF → llama-server, ONNX speech-to-text → python -m drove.workers.asr,
                    .tts text-to-speech → uv run --python 3.12 … workers/tts_runtime/worker.py)
               → wait for backend /health
           → httpx reverse proxy → backend port
           → reset inactivity timer
```

## Core modules

- `src/drove/proxy.py` — OpenAI-compatible reverse proxy
- `src/drove/server_manager.py` — lifecycle management for backend subprocesses
- `src/drove/backend.py` — picks the backend for a model (llama-server, ASR worker, or TTS worker)
- `src/drove/workers/asr.py` — built-in ONNX speech-to-text worker
- `src/drove/workers/tts_runtime/` — text-to-speech worker (`worker.py`), the Kokoro, Qwen3-TTS,
  Chatterbox, Fish, and KittenTTS 2 engines (`engines.py`), and one pinned requirements file per engine
- `src/drove/config.py` — global config loading + defaults
- `src/drove/model_config.py` — per-model config parsing
- `src/drove/cli/` — Typer CLI

## Process model

- A model process is started lazily when the first request arrives.
- The backend per model depends on its file type: `.gguf` models run `llama-server`,
  `.onnx` speech-to-text models run the built-in ASR worker, and `.tts` registrations run
  the text-to-speech worker. An explicit `backend` key in the sidecar overrides the file type.
- The TTS engines need PyTorch versions capped below drove's Python 3.14, so the TTS worker
  runs under `uv run --no-project --python 3.12 --with-requirements <engine>.txt`. uv builds
  and caches one environment per engine on first start; drove's own environment is untouched.
  The worker runs in its own process group so the whole tree is stopped on idle shutdown.
- Every backend shares the same health check, idle timeout, LRU eviction, and memory
  budget. Only llama-server models use the prompt (KV) cache.
- `drove` keeps request-path behavior transparent by forwarding requests directly.
- After inactivity timeout, the backend process is terminated to release resources.
