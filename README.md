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

- **Lazy model loading** — models start on first request and shut down after inactivity
- **Multiple concurrent models** — run several models simultaneously, each on its own llama-server instance
- **HuggingFace integration** — download models directly with `vllama models download`
- **Auto-conversion** — safetensors models are automatically converted to GGUF on download
- **Per-model configuration** — set context size, GPU layers, and other parameters per model
- **Interactive chat** — TUI chat interface with session persistence, collapsible thinking, and token speed display
- **Any OpenAI endpoint** — chat with local models or connect to any OpenAI-compatible API (OpenAI, Groq, etc.)
- **OpenAI-compatible API** — full proxy with `/v1/chat/completions`, `/v1/models`, and server stats
- **Observability** — log all API requests/responses to disk and browse them in a TUI
- **Live metrics** — token speed (tok/s), TTFT, and usage stats via `vllama server status`
- **Systemd support** — optional user service for persistent operation

## Commands

### `vllama chat`

Open an interactive TUI chat session. Connects to the local vllama server by default, or to any OpenAI-compatible API with `--endpoint`.

```bash
# Chat with the local server (prompts you to pick a model)
vllama chat

# Specify a model directly
vllama chat Qwen3.5-35B-A3B-Q4_K_M

# Resume the latest session for the selected model
vllama chat --resume

# Connect to any OpenAI-compatible endpoint
vllama chat --endpoint https://api.openai.com/v1 --api-key sk-...
vllama chat -e https://api.groq.com/openai/v1 -k gsk-...
```

When no model is specified, vllama queries the endpoint's `/v1/models` and prompts you to pick one. Use `/help` inside the chat to see available commands (`/sessions`, `/theme`, etc.).

**Arguments:**

| Argument | Description |
|----------|-------------|
| `model`  | Model name to chat with (optional, prompts if omitted) |

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--endpoint` | `-e` | OpenAI-compatible base URL (e.g. `https://api.openai.com/v1`) |
| `--api-key` | `-k` | API key for the endpoint |
| `--system` | `-s` | Set a system prompt |
| `--resume` | `-r` | Resume the latest saved session |
| `--host` | | Override local server host |
| `--port` | | Override local server port |

---

### `vllama server`

Start the vllama proxy server. Models are started lazily on first API request and shut down individually after the configured idle timeout.

```bash
# Start with defaults from config
vllama server

# Override host and port
vllama server --host 127.0.0.1 --port 9090
```

Multiple models can run concurrently — each gets its own `llama-server` process on a separate port, managed transparently by the proxy.

**Options:**

| Option | Description |
|--------|-------------|
| `--host` | Listen host (overrides config) |
| `--port` | Listen port (overrides config) |

---

### `vllama server status`

Show the status of the running vllama server, including loaded models, resource usage, request counts, and token throughput.

```bash
# Show status once
vllama server status

# Continuously refresh every 2 seconds (default)
vllama server status --watch

# Refresh every 5 seconds
vllama server status --watch 5
```

Example output:

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

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--watch` | `-w` | Continuously refresh every N seconds (default 2 if no value given) |

---

### `vllama observe`

Browse logged API requests and responses in a TUI. Enable observation logging first by setting `observe = true` in your config:

```bash
# Enable logging
vllama config observe true

# Browse all logged requests
vllama observe

# Filter by model
vllama observe Qwen3.5-35B-A3B-Q4_K_M
```

The TUI shows a list of requests on the left and a detail pane on the right. Select a request to inspect its headers, request body, and response. Streaming responses (SSE) are automatically assembled into readable output — the full content, reasoning, and tool calls are combined into a single JSON view. The raw SSE stream is available in a collapsed "Raw Response" section for advanced debugging.

Logs are stored in `~/.local/share/vllama/observe/` (configurable via `observe_dir`).

**Arguments:**

| Argument | Description |
|----------|-------------|
| `model`  | Filter logs by model name (optional) |

**Key bindings:**

| Key | Action |
|-----|--------|
| Arrow keys | Navigate request list |
| `r` | Refresh the list |
| `q` / `Escape` | Quit |

**Configuration keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `observe` | `false` | Enable request/response logging |
| `observe_dir` | `~/.local/share/vllama/observe` | Log storage directory |

---

### `vllama models download`

Download a model from HuggingFace Hub.

```bash
# Download a GGUF model (prompts to select quantization if multiple available)
vllama models download unsloth/Qwen3-8B-GGUF

# Specify quantization directly
vllama models download "unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M"

# Override the local model name
vllama models download unsloth/Qwen3-8B-GGUF --name my-qwen

# Skip confirmation prompt
vllama models download unsloth/Qwen3-8B-GGUF --yes
```

The format is `org/repo` or `org/repo:QUANT`. When a quantization tag is provided (e.g. `:Q4_K_M`), only matching files are downloaded. Sharded models are stored in a named subdirectory. Download metadata (repo, files, quantization) is saved to a sidecar TOML.

**Arguments:**

| Argument | Description |
|----------|-------------|
| `model_ref` | HuggingFace repo reference (`org/repo` or `org/repo:QUANT`) |

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--name` | `-n` | Override local model name |
| `--yes` | `-y` | Skip confirmation prompt |

---

### `vllama models list`

List all downloaded models.

```bash
# List models
vllama models list

# Show download origin info
vllama models list --verbose
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-V` | Show download origin info (HuggingFace repo, files) |

---

### `vllama models info`

Show info and configuration for a specific model.

```bash
vllama models info Qwen3.5-35B-A3B-Q4_K_M
```

Displays the model name, file count, total size, primary file path, download origin, and any model-specific parameters.

**Arguments:**

| Argument | Description |
|----------|-------------|
| `name` | Model name |

---

### `vllama models config`

Get or set per-model configuration parameters. These map directly to `llama-server` CLI flags (snake_case keys become `--kebab-case` flags).

```bash
# Show all parameters for a model
vllama models config mymodel

# Get a single parameter
vllama models config mymodel ctx_size

# Set a parameter
vllama models config mymodel ctx_size 8192

# Remove a parameter
vllama models config mymodel --unset ctx_size

# Operate on global model defaults instead
vllama models config --global
vllama models config --global ctx_size 8192
vllama models config --global --unset ctx_size
```

Per-model settings override global settings. Global config is stored in `_global.toml` in the models directory.

**Arguments:**

| Argument | Description |
|----------|-------------|
| `name` | Model name (omit when using `--global`) |
| `key` | Config key to get or set (optional) |
| `value` | Value to set (optional) |

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--global` | `-g` | Operate on global model defaults |
| `--unset` | | Remove a config key |

**Available parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx_size` | int | Context window size |
| `n_gpu_layers` | int | Number of layers to offload to GPU |
| `main_gpu` | int | Main GPU to use |
| `tensor_split` | string | GPU tensor split |
| `batch_size` | int | Batch size |
| `ubatch_size` | int | Ubatch size |
| `n_parallel` | int | Number of parallel sequences |
| `temp` | float | Temperature |
| `top_p` | float | Top-p nucleus sampling |
| `top_k` | int | Top-k sampling |
| `threads` | int | Number of CPU threads |
| `threads_batch` | int | Threads for batch processing |
| `flash_attn` | bool | Enable flash attention |
| `rope_freq_base` | float | RoPE frequency base |
| `rope_freq_scale` | float | RoPE frequency scale |
| `cache_type_k` | string | K cache quantization type |
| `cache_type_v` | string | V cache quantization type |

---

### `vllama models delete`

Delete a model and its config.

```bash
vllama models delete Qwen3.5-35B-A3B-Q4_K_M

# Skip confirmation
vllama models delete Qwen3.5-35B-A3B-Q4_K_M --yes
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `name` | Model name |

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--yes` | `-y` | Skip confirmation prompt |

---

### `vllama config`

Show or edit global configuration values.

```bash
# Show all config values
vllama config

# Get a single value
vllama config idle_timeout_seconds

# Set a value
vllama config idle_timeout_seconds 3600

# Set a nested value
vllama config llama_server.n_gpu_layers 42
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `key` | Config key to get or set (optional, shows all if omitted) |
| `value` | Value to set (optional, gets value if omitted) |

**Configuration keys:**

| Key | Default | Description |
|-----|---------|-------------|
| `models_dir` | `~/.local/share/vllama/models` | Model storage directory |
| `sessions_dir` | `~/.local/share/vllama/sessions` | Chat session storage |
| `observe` | `false` | Enable request/response logging |
| `observe_dir` | `~/.local/share/vllama/observe` | Observe log storage |
| `listen_host` | `0.0.0.0` | Server listen host |
| `listen_port` | `8080` | Server listen port |
| `llama_server_bin` | `llama-server` | Path to llama-server binary |
| `idle_timeout_seconds` | `1800` | Idle shutdown timeout (30 min) |
| `llama_server_host` | `127.0.0.1` | llama-server bind host |
| `tui_theme` | `textual-dark` | TUI color theme |
| `llama_server.n_gpu_layers` | `-1` | Default GPU layers (all) |
| `llama_server.threads` | `null` | Default CPU threads |

Environment variables with `VLLAMA_*` prefix override config file values.

---

### `vllama init`

Create the config file at its default location with all default values.

```bash
# Create config
vllama init

# Overwrite existing config
vllama init --force
```

Writes to `~/.config/vllama/config.toml` (or the path from `--config` / `VLLAMA_CONFIG` env var).

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--force` | `-f` | Overwrite existing config |

---

### `vllama completions`

Manage shell completions for tab-completion of commands, model names, and options.

**Generate a completion script:**

```bash
vllama completions generate zsh > ~/.zfunc/_vllama
vllama completions generate bash | sudo tee /etc/bash_completion.d/vllama
```

**Install completions automatically:**

```bash
# Auto-detect shell and install
vllama completions install

# Specify shell explicitly
vllama completions install zsh

# Preview without writing files
vllama completions install --dry-run
```

**List supported shells:**

```bash
vllama completions shells
```

Supported shells: `bash`, `zsh`, `fish`, `powershell`.

## Global Options

These options are available on all commands:

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to config TOML file |
| `--verbose` | `-v` | Enable debug logging |

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

Use the endpoint URL shown by `vllama server status` as `OPENAI_BASE_URL` in any OpenAI-compatible client.

## Quick Start

```bash
# Initialize config
vllama init

# Download a model
vllama models download unsloth/Qwen3-8B-GGUF

# Start the proxy server
vllama server

# Chat interactively
vllama chat
```

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
