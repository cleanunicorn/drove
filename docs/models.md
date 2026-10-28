# Managing models

[Docs](./README.md) / Managing models

Download a model, find the name clients should use, and tune its runtime
settings. Run [Getting started](./getting-started.md) first if drove is not
installed.

## Download and name a model

```bash
drove models download unsloth/Qwen3-8B-GGUF:Q4_K_M --name qwen
```

A download reference is `org/repo` or `org/repo:QUANT`. Without a quantization
tag, repositories with several variants show a picker. Partial downloads
resume; rerun the same command after an interrupted transfer. Sharded weights
are kept together, and multimodal projectors are detected automatically.

drove downloads GGUF weights for text and vision models and ONNX weights for
speech-to-text; text-to-speech repositories keep their engine's own files. It
never converts weights: a repository that only holds safetensors, `.bin` or
`.pt` files is refused before anything is downloaded. Look for a GGUF build of
the model instead, often published as a separate `-GGUF` repository.

An interrupted download is not listed, offered to agents or served until it
finishes. Each download records its expected files in a hidden
`.drove-download.json` and is marked complete only after the last file lands;
a later missing or truncated file, or a split GGUF without all its shards,
hides the model again. `drove models info <name>` explains what is missing.
Rerun the download to finish it, or delete it. Models downloaded before this
record existed are trusted as they are, except that split GGUFs still need
every shard.

`--name qwen` gives the download a stable local name. Without `--name`, the name
is derived from the repository and selected quantization, such as
`unsloth/Qwen3-8B-GGUF:Q4_K_M`. Always check the result:

```bash
drove models list
drove models info qwen
```

A local name is `name` or `org/name`. Each part starts with a letter or digit
and may contain letters, digits, `.`, `_`, `:` and `-`, so
`unsloth/Qwen3-8B-GGUF:Q4_K_M` is valid. Names with spaces, `.` or `..`
parts, backslashes, a leading `/`, or more than one `/` are rejected by every
command and by the API. Existing models whose file or directory name breaks this
rule are hidden from `drove models list`, completion and `GET /v1/models` until
you rename them on disk. Model files must also stay inside `models_dir`: a model
directory or file that is a symlink to somewhere outside it is not listed or
served. To keep models on another disk, point `models_dir` at that disk (a
symlinked `models_dir` itself is fine) rather than linking single models in.

Use the exact `NAME` from the list as the API `model` value or chat argument.
The server's `GET /v1/models` returns the same local names as `id` values.
Repository aliases can resolve to downloaded variants, but explicit local names
avoid ambiguity when you keep more than one quantization.

Downloads live under `~/.local/share/drove/models/` by default. Set `models_dir`
in the [global configuration](./configuration.md) to use another disk.

## Choosing a download variant

When a GGUF or ONNX repository has several variants, use ↑/↓ and Enter to
choose. The picker uses your `tui_theme` and shows each variant's full download
size (weights plus companion files), whether it is fully or partly downloaded,
and an advisory memory estimate. Unknown metadata is shown explicitly.

The estimate compares weights and projector size with 80% of available host
RAM. It is not a guarantee: context size, KV cache, runtime overhead and GPU
placement affect actual requirements. A fitting `Q4_K_M` (GGUF) or `int8`
(ONNX) is preferred; otherwise the smallest known fitting variant is selected.
Oversized variants remain selectable. With no memory information, the picker
shows that limitation instead of claiming a fit.

**Download every variant** is separated below the individual variants and
shows the total with shared files counted once. Files found in a different
all-variants directory are labeled with that location; they are not silently
moved or reused in a new quantization directory. Esc or Ctrl+C cancels without
downloading. The usual download/resume confirmation follows selection.

For scripts or redirected output, pass `org/repo:QUANT` to bypass the picker.
`--yes` never downloads every GGUF quantization: on a repository with several,
it stops and asks for `org/repo:QUANT` without downloading anything. For ONNX
it selects the default variant. TTS and single-variant repositories skip the
picker.

## Inspect what is installed

| Command | Use it to… |
| --- | --- |
| `drove models list` | See names, capabilities, size, and total disk use |
| `drove models list -V` | Include storage paths and download origins |
| `drove models info qwen` | Inspect capabilities, quantization, and explicit overrides |
| `drove models info qwen -V` | Include local files and paths |
| `drove models config qwen` | See effective settings and the source of each value |
| `drove models config qwen --all` | Include unset model parameters |

Capabilities include `vision`, `stt` (speech-to-text), and `tts`
(text-to-speech). Listing or inspecting a model does not load it into memory.

## Set runtime parameters

Set an override for one model:

```bash
drove models config qwen ctx_size 8192
drove models config qwen n_gpu_layers -1
```

Set a default shared by all models, then remove an individual override to
inherit it:

```bash
drove models config --global ctx_size 16384
drove models config qwen --unset ctx_size
drove models config qwen
```

Effective values are resolved in this order, highest priority first:

1. The model's sidecar TOML file.
2. `_global.toml` in the models directory.
3. Supported defaults in `config.toml` under `[llama_server]` (`n_gpu_layers`, `threads`).
4. The backend's own defaults for unset parameters.

A sidecar sits beside the primary model file with its extension changed to
`.toml`; for example, `model.gguf` uses `model.toml`. Downloaded models live in
directories, while legacy flat `qwen.gguf` files use a sibling `qwen.toml`.
Prefer the CLI so you do not need to locate the primary file yourself. Changes to startup arguments
apply the next time the backend starts; a running process keeps its current
arguments. Stop and restart the proxy to apply them immediately.

Only declared model keys are recognized; unknown TOML keys are ignored.
For a llama-server flag drove does not expose, use `extra_args`:

```bash
drove models config qwen extra_args "--no-webui"
```

See [Configuration](./configuration.md#per-model-config) for the supported keys
and [resource limits](./configuration.md#model-eviction) for memory budgeting.

## Speech models

- [Speech-to-text](./speech-to-text.md) uses ONNX exports and the `asr` extra.
- [Text-to-speech](./text-to-speech.md) detects supported engines on download,
  or registers them with `drove models add-tts` for first-use downloads.

They share the proxy's idle timeout and eviction limits with text models.

## Delete a model

```bash
drove models delete qwen
```

The command asks for confirmation before removing local model files and their
configuration. `--yes` skips the prompt. Do not delete a model while it is
serving requests; stop the proxy first. After changing your inventory, rerun
`drove integrate <agent>` if you use a coding agent.

[Docs](./README.md) · [Text and vision](./text-and-vision.md) · [Configuration](./configuration.md)
