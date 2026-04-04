# vllama

llama.cpp server manager, proxy and model downloader.

Start `llama-server` on demand, proxy OpenAI-compatible API requests, and shut it down after inactivity. Similar to `ollama` but wraps `llama-server` directly.

## Prerequisites

[`llama.cpp`](https://github.com/ggml-org/llama.cpp/) must be installed and `llama-server` must be available in your PATH.

Verify installation:
```bash
llama-server --version
```

## Features

- **Lazy model loading**: Models start on first request and shut down after inactivity
- **Multiple concurrent models**: Run several models simultaneously, each on its own llama-server instance
- **HuggingFace integration**: Download models directly with `vllama models download`
- **Auto-conversion**: Safetensors models are automatically converted to GGUF on download
- **Per-model configuration**: Set context size, GPU layers, and other parameters per model
- **Interactive chat**: TUI chat interface with session persistence, collapsible thinking, and token speed display
- **Any OpenAI endpoint**: Chat with local models or connect to any OpenAI-compatible API (OpenAI, Groq, etc.)
- **OpenAI-compatible API**: Full proxy with `/v1/chat/completions`, `/v1/models`, and server stats
- **Live metrics**: Token speed (tok/s), TTFT, and usage stats via `vllama status`
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

## Systemd Service (Optional)

Run vllama as a background user service:

```bash
# Install and start service
make service-install

# Manage service
make service-start
make service-stop
make service-status
make service-logs

# Remove service
make service-uninstall
```

To enable at boot without login:
```bash
loginctl enable-linger $USER
```

## Installation

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
vllama serve
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
# Chat with the local vllama server (lists available models)
vllama chat

# Specify a model directly
vllama chat Qwen3.5-35B-A3B-Q4_K_M

# Resume the latest session
vllama chat --resume

# Connect to any OpenAI-compatible endpoint
vllama chat --endpoint https://api.openai.com/v1 --api-key sk-...
vllama chat -e https://api.groq.com/openai/v1 -k gsk-...
```

Options:
- `--endpoint`, `-e` - OpenAI-compatible base URL (connects to local server if omitted)
- `--api-key`, `-k` - API key for the endpoint
- `--system`, `-s` - Set system prompt
- `--resume`, `-r` - Resume latest session
- `--host`, `--port` - Override local server address

When no model is specified, vllama queries the endpoint's `/v1/models` and prompts you to pick one.

### Status

Show server status with token speed and usage stats:

```bash
vllama server status
```

```
Server:    running (8m 53s uptime)
Listen:    0.0.0.0:7777
Endpoint:  http://127.0.0.1:7777/v1
Models:    2 loaded
  - gemma-4-E4B-it-Q4_K_M
    Loaded:  5m 12s ago
    Idle:    26s / 10m 0s
  - Hermes-3-Llama-3.2-3B.Q8_0
    Loaded:  4m 52s ago
    Idle:    54s / 10m 0s
    Active:  1 request(s)
Process (gemma-4-E4B-it-Q4_K_M): 3.1 GB RSS, 12.1% CPU
Process (Hermes-3-Llama-3.2-3B.Q8_0): 2.1 GB RSS, 222.5% CPU
Requests:  10 total, 1 active, 0 errors
Tokens:    1218 in / 3237 out (4455 total)
Speed:     86.5 tok/s (last), 89.8 tok/s (avg)
TTFT:      0.000s (last), 0.000s (avg)
```

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
# List available models
curl http://localhost:8080/v1/models

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b", "messages": [{"role": "user", "content": "Hello!"}]}'

# Server status and metrics
curl http://localhost:8080/status
```

Models are started lazily on first request and shut down after `idle_timeout_seconds` of inactivity.

Use the endpoint URL shown by `vllama status` as `OPENAI_BASE_URL` in any OpenAI-compatible client.

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
