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
max_loaded_models = 1
max_memory = "24GB"

[llama_server]
n_gpu_layers = -1
```

## Model eviction

Two independent limits control when loaded models are stopped to make room
for a newly requested one (in both cases the least-recently-used idle model
is evicted first; models with in-flight requests are drained before being
stopped):

- `max_loaded_models` — how many models may be loaded at once (`0` = unlimited, default `1`).
- `max_memory` — combined memory budget for all loaded models (`"0"` = unlimited, the default).
  Accepts decimal (`"24GB"`, `"512MB"`) and binary (`"16GiB"`) units, or a plain
  number of bytes.

The memory used by a model is estimated from its on-disk file size (all shards
for sharded GGUF models, all `.onnx` files for speech-to-text models). Context
(KV cache) and runtime overhead are not counted, so leave some headroom below
your real RAM/VRAM limit. A model whose estimate alone exceeds `max_memory` is
still started (after evicting everything else) rather than refused.

## Per-model config

Each model can have a sidecar config file in the models directory:

- model file: `~/.local/share/drove/models/<name>.gguf`
- config file: `~/.local/share/drove/models/<name>.toml`

Example:

```toml
ctx_size = 4096
n_gpu_layers = -1
```

Only keys declared in `ModelConfig` are accepted; unknown keys are silently ignored.
Supported keys: `ctx_size`, `n_gpu_layers`, `main_gpu`, `tensor_split`, `batch_size`,
`ubatch_size`, `n_parallel`, `temp`, `top_p`, `top_k`, `threads`, `threads_batch`,
`flash_attn`, `rope_freq_base`, `rope_freq_scale`, `cache_type_k`, `cache_type_v`, `mmproj`.

Drove-specific keys (never passed to llama-server): `backend` (`llama` or `asr`),
`asr_model`, `asr_quantization` — see [Speech-to-text](./speech-to-text.md).
