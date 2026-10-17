# CLI reference

## Core commands

```bash
drove --help
drove init
drove serve
drove chat
```

## Config management

```bash
drove config
drove config idle_timeout_seconds
drove config idle_timeout_seconds 3600
drove config llama_server.n_gpu_layers -1
```

## Models management

```bash
drove models --help
drove models list
drove models download <repo>
drove models download hexgrad/Kokoro-82M --name kokoro
drove models delete <name>
drove models info <name>
drove models config <name>
```

The `CAPS` column in `drove models list` shows model capabilities:
`vision` (multimodal projection), `stt` (speech-to-text), and `tts` (text-to-speech).
TTS downloads preserve voice/tokenizer directories and configure the speech worker
automatically; see [Text-to-speech](text-to-speech.md).

## Coding agent integration

```bash
drove integrate <agent> [MODEL ...]
```

Registers drove models with a local coding agent by writing a `drove`
provider into the agent's config file. Supported agents:

| Agent | Config file written |
|--|--|
| `pi` | `~/.pi/agent/models.json` (`$PI_CODING_AGENT_DIR`); `--default` goes to the `settings.json` beside it |
| `opencode` | `~/.config/opencode/opencode.json` (or an existing `opencode.jsonc` / `config.json`; `$OPENCODE_CONFIG`) |
| `kilo` | `~/.config/kilo/kilo.json` (or an existing `kilo.jsonc` / `opencode.json[c]` / `config.json`; `$KILO_CONFIG`) |

Each model entry carries its full setup:

- **context window** — `ctx_size` (or `-c` in `extra_args`) from the model's
  sidecar config, then `_global.toml`; `0` or unset means the GGUF training
  context. It is the context one request can use: divided across slots when
  `n_parallel` is set without a unified KV cache, limited by
  `--kv-unified-per-slot`, and capped at the training context;
- **output limit** — a quarter of the context, between 1,024 and 32,768
  tokens, and never more than half the context;
- **vision** — an `mmproj` projector is present and not disabled with
  `--no-mmproj`;
- **tool calling** and **reasoning** — read from the chat template llama-server
  will use (the GGUF one, or a `--chat-template[-file]` override); `--no-jinja`
  turns both off and `--reasoning-budget 0` turns reasoning off.

With no model names, every chat model is added (speech-to-text models are
skipped) and models drove no longer has are removed from the provider. A
removed model is also cleared from wherever it was selected (opencode/Kilo
`model` and `small_model`, pi `defaultModel` and `enabledModels`). Keys you
added to a model entry by hand are kept, at any depth. Only files that change
are written, all or none; the previous version is saved as `<file>.bak`, and
comments in a `.jsonc` file are not preserved.

| Option | Meaning |
|--|--|
| `-d, --default MODEL` | Also make this model the agent's default. |
| `--remove` | Remove the given models, or the whole provider if none are given. |
| `--base-url URL` | URL the agent uses (default `http://<listen_host>:<listen_port>/v1`). |
| `--provider-id ID` | Provider name in the agent config (default `drove`). |
| `-f, --file PATH` | Edit this config file instead of the auto-detected one. |
| `-n, --dry-run` | Print the resulting config without writing it. |

## Shell completions

```bash
drove completions --help
```
