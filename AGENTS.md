# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`drove` is a llama.cpp server manager/proxy — similar to ollama but wrapping `llama-server` directly. It lazily starts `llama-server` on the first request for a model, proxies OpenAI-format API traffic to it, and shuts it down after a configurable inactivity period.

## Development Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management, with a `Makefile` wrapping the common tasks. **Always use the `make` targets rather than invoking `uv`/`pytest`/`ruff`/`mypy` directly.**

```bash
# Install drove (and its dependencies) as a uv tool
make install

# Run tests
make test

# Lint
make lint

# Format
make fmt

# Type check
make typecheck

# Install shell completions (also runs make install)
make completions
```

Note that `make install` installs the `drove` CLI globally (via `uv tool install`) so you can *run* it — it is not a prerequisite for development. `make test`, `make lint`, `make fmt`, and `make typecheck` use `uv run` against the project environment directly, which uv syncs automatically, so they work without `make install`.

After `make install`, run the server and CLI via the installed `drove` binary:

```bash
# Run the server
drove serve

# Run the CLI
drove <command>
```

For running an individual test file or a single test by name, there is no
dedicated `make` target, so fall back to `pytest` through uv:

```bash
# Run a single test file
uv run pytest tests/test_proxy.py

# Run a single test by name
uv run pytest tests/test_proxy.py::test_name
```

A systemd user service can be managed with the `service-*` targets
(`make service-install`, `make service-start`, `make service-stop`,
`make service-restart`, `make service-status`, `make service-logs`,
`make service-uninstall`).

## Architecture

### Key Components

**`src/drove/config.py`** — Global config via `pydantic-settings`. Settings are loaded from `~/.config/drove/config.toml` (or path from `DROVE_CONFIG` env var), with env var overrides prefixed `DROVE_*`.

**`src/drove/model_config.py`** — Per-model config (context size, GPU layers, etc.) stored as TOML files alongside model weights in the models directory. Loaded and merged into `llama-server` CLI args.

**`src/drove/server_manager.py`** — Manages backend subprocess lifecycles (one per model): start, stop, health check, inactivity timer, LRU eviction. Uses `asyncio.subprocess`. The backend per model is chosen by `src/drove/backend.py`: GGUF models run `llama-server`, ONNX speech-to-text models run the built-in ASR worker.

**`src/drove/workers/asr.py`** — Built-in speech-to-text worker spawned as `python -m drove.workers.asr`. Loads ONNX ASR models (e.g. NVIDIA Parakeet) via the optional `onnx-asr` package (`drove[asr]` extra) and serves an OpenAI-compatible `/v1/audio/transcriptions` plus `/health`.

**`src/drove/proxy.py`** — FastAPI app that acts as a reverse proxy. On each request it extracts the model name (from JSON or multipart form bodies), calls `ServerManager.ensure_running(model)`, then forwards the request to the backend via `httpx.AsyncClient`. Resets the inactivity timer on each proxied request.

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

## Changelog

Every change to this repository **must** be recorded in `CHANGELOG.md` under the `## [Unreleased]` section before the work is committed. No exceptions — bug fixes, features, refactors, doc updates, and dependency bumps all get an entry.

- Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Group entries under `### Added`, `### Changed`, `### Deprecated`, `### Removed`, `### Fixed`, or `### Security` (create the subsection if it doesn't exist yet under `[Unreleased]`).
- Write one bullet per user-visible change, in past tense, describing the impact rather than the implementation. Reference the affected module or CLI command when useful.
- Do **not** assign a version number or release date — the release automation moves `[Unreleased]` entries into a new versioned section when it cuts a release.
- Versioning is driven by [Conventional Commits](https://www.conventionalcommits.org/) on merge to `main` (`fix:` → patch, `feat:` → minor, `feat!:` / `BREAKING CHANGE:` → major). The changelog entry must be consistent with the commit type.
