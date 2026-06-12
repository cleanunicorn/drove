# Architecture

## Request flow

```text
Client → FastAPI proxy (drove port)
           → ServerManager.ensure_running(model)
               → if not running: spawn backend subprocess
                   (GGUF → llama-server, ONNX speech-to-text → python -m drove.workers.asr)
               → wait for backend /health
           → httpx reverse proxy → backend port
           → reset inactivity timer
```

## Core modules

- `src/drove/proxy.py` — OpenAI-compatible reverse proxy
- `src/drove/server_manager.py` — lifecycle management for backend subprocesses
- `src/drove/backend.py` — picks the backend for a model (llama-server vs ASR worker)
- `src/drove/workers/asr.py` — built-in ONNX speech-to-text worker
- `src/drove/config.py` — global config loading + defaults
- `src/drove/model_config.py` — per-model config parsing
- `src/drove/cli/` — Typer CLI

## Process model

- A model process is started lazily when the first request arrives.
- The backend per model depends on its file type: `.gguf` models run `llama-server`,
  `.onnx` speech-to-text models run the built-in ASR worker.
- `drove` keeps request-path behavior transparent by forwarding requests directly.
- After inactivity timeout, the backend process is terminated to release resources.
