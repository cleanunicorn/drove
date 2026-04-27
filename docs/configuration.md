# Configuration

`drove` reads global settings from:

- `~/.config/drove/config.toml`
- or a custom path set with `DROVE_CONFIG`

Environment variables with the `DROVE_` prefix override file values.

## Example config

```toml
models_dir = "~/.local/share/drove/models"
listen_host = "0.0.0.0"
listen_port = 8080
llama_server_bin = "llama-server"
idle_timeout_seconds = 1800

[llama_server]
n_gpu_layers = -1
```

## Per-model config

Each model can have a sidecar config file in the models directory:

- model file: `~/.local/share/drove/models/<name>.gguf`
- config file: `~/.local/share/drove/models/<name>.toml`

Example:

```toml
context_size = 4096
n_gpu_layers = -1
```

Any `llama-server` flag can be represented as a `snake_case` key.
