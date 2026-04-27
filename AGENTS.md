# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`drove` is a llama.cpp server manager/proxy — similar to ollama but wrapping `llama-server` directly. It lazily starts `llama-server` on the first request for a model, proxies OpenAI-format API traffic to it, and shuts it down after a configurable inactivity period.

## Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
# Install dependencies and create venv
uv sync

# Run the server
uv run drove serve

# Run the CLI
uv run drove <command>

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_proxy.py

# Run a single test by name
uv run pytest tests/test_proxy.py::test_name

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy src/
```

## Architecture

### Key Components

**`src/drove/config.py`** — Global config via `pydantic-settings`. Settings are loaded from `~/.config/drove/config.toml` (or path from `DROVE_CONFIG` env var), with env var overrides prefixed `DROVE_*`.

**`src/drove/model_config.py`** — Per-model config (context size, GPU layers, etc.) stored as TOML files alongside model weights in the models directory. Loaded and merged into `llama-server` CLI args.

**`src/drove/server_manager.py`** — Manages the `llama-server` subprocess lifecycle: start, stop, health check, inactivity timer. Only one model runs at a time. Uses `asyncio.subprocess`.

**`src/drove/proxy.py`** — FastAPI app that acts as a reverse proxy. On each request it calls `ServerManager.ensure_running(model)`, then forwards the request to `llama-server` via `httpx.AsyncClient`. Resets the inactivity timer on each proxied request.

**`src/drove/cli/`** — Typer CLI with subcommands: `serve`, `models list`, `models download`, `models delete`, `models info`, `models config`.

### Request Flow

```
Client → FastAPI proxy (drove port)
           → ServerManager.ensure_running(model)
               → if not running: spawn llama-server subprocess
               → wait for llama-server /health
           → httpx reverse proxy → llama-server port
           → reset inactivity timer
```

### Model Storage

Models are stored in a flat directory (default `~/.local/share/drove/models/`). The filename (without extension) is the model name. Each model can have a sidecar config file `<model_name>.toml` in the same directory.

### Inactivity Shutdown

`ServerManager` runs an `asyncio` background task that checks last-request time and calls `llama-server` SIGTERM after the configured idle timeout (default 30 min).

## Config File Format

```toml
# ~/.config/drove/config.toml
models_dir = "~/.local/share/drove/models"
listen_host = "0.0.0.0"
listen_port = 8080
llama_server_bin = "llama-server"
idle_timeout_seconds = 1800

[llama_server]
# default llama-server args applied to all models
n_gpu_layers = -1
```

## Per-Model Config Format

```toml
# ~/.local/share/drove/models/<name>.toml
context_size = 4096
n_gpu_layers = -1
# any llama-server flag as snake_case key
```
