# Architecture

## Request flow

```text
Client → FastAPI proxy (drove port)
           → ServerManager.ensure_running(model)
               → if not running: spawn llama-server subprocess
               → wait for llama-server /health
           → httpx reverse proxy → llama-server port
           → reset inactivity timer
```

## Core modules

- `src/drove/proxy.py` — OpenAI-compatible reverse proxy
- `src/drove/server_manager.py` — lifecycle management for `llama-server`
- `src/drove/config.py` — global config loading + defaults
- `src/drove/model_config.py` — per-model config parsing
- `src/drove/cli/` — Typer CLI

## Process model

- A model process is started lazily when the first request arrives.
- `drove` keeps request-path behavior transparent by forwarding requests directly.
- After inactivity timeout, `llama-server` is terminated to release resources.
