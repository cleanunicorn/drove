# vllama

llama.cpp server manager and proxy.

Start `llama-server` on demand, proxy OpenAI-compatible API requests, and shut it down after inactivity. Similar to `ollama` but wraps `llama-server` directly.

## Prerequisites

[`llama.cpp`](https://github.com/ggml-org/llama.cpp/) must be installed and `llama-server` must be available in your PATH.

Verify installation:
```bash
llama-server --version
```

## Features

- **Lazy model loading**: Models start on first request and shut down after inactivity
- **HuggingFace integration**: Download models directly with `vllama models download`
- **Per-model configuration**: Set context size, GPU layers, and other parameters per model
- **Interactive chat**: TUI chat interface with session persistence
- **OpenAI-compatible API**: Proxy requests to llama-server with standard endpoints
- **Systemd support**: Optional user service for persistent operation

## Model Management

Models are stored in `~/.local/share/vllama/models/`. Each model can have a sidecar config file `<name>.toml`.

**Download:**
```bash
vllama models download unsloth/Qwen3-8B-GGUF
vllama models download "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M"
```

**List:** `vllama models list`

**Info:** `vllama models info <name>`

**Delete:** `vllama models delete <name>`

**Configure:** `vllama models config <name> ctx_size 8192`

```bash
# Install via uv tool
uv tool install vllama

# Or install from source
make install
```

## Quick Start

```bash
# Initialize config
vllama init

# Download a model
vllama models download unsloth/Qwen3-8B-GGUF

# Start the proxy server
vllama serve

# Chat interactively
vllama chat
```

## Commands

### Serve

Start the vllama proxy server:

```bash
uv run vllama serve
```

Options:
- `--host` - Listen host (default from config)
- `--port` - Listen port (default from config)

### Models

**List models:**
```bash
vllama models list
```

**Download a model:**
```bash
vllama models download unsloth/Qwen3-5-8B-GGUF
vllama models download "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M"
```

Format: `org/repo[:quantization]`

**Model info:**
```bash
vllama models info <model-name>
```

**Delete a model:**
```bash
vllama models delete <model-name>
```

**Configure model:**
```bash
vllama models config <model-name>          # show all params
vllama models config <model-name> ctx_size # get one param
vllama models config <model-name> ctx_size 8192  # set a param
vllama models config <model-name> --unset ctx_size  # remove param
```

### Config

**Show config:**
```bash
vllama config
```

**Get/set values:**
```bash
vllama config idle_timeout_seconds              # get value
vllama config idle_timeout_seconds 3600         # set value
vllama config llama_server.n_gpu_layers 42      # nested key
```

### Chat

Open an interactive TUI chat session:

```bash
vllama chat
vllama chat --resume
```

Options:
- `--system`, `-s` - Set system prompt
- `--resume`, `-r` - Resume latest session
- `--host`, `--port` - Override server address

## Configuration

Config file: `~/.config/vllama/config.toml`

```toml
models_dir = "~/.local/share/vllama/models"
listen_host = "0.0.0.0"
listen_port = 8080
llama_server_bin = "llama-server"
idle_timeout_seconds = 1800
tui_theme = "dracula"

[llama_server]
n_gpu_layers = -1
```

Environment variables with `VLLAMA_*` prefix override config.

### Per-Model Config

Store model-specific settings in `<model-name>.toml` alongside the weights:

```toml
context_size = 4096
n_gpu_layers = 42
```

## API

vllama proxies to llama-server and exposes an OpenAI-compatible API. Once the server is running:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Models are started lazily on first request and shut down after `idle_timeout_seconds` of inactivity.

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy src/
```
