# CLI reference

[Docs](./README.md) / CLI reference

Use `drove <command> --help` for the complete option list. Root options precede
the command; subcommand options follow it.

[Commands](#core-commands) · [Chat](#chat-model-selection) · [Config](#config-management) · [Models](#models-management) ·
[Agents](#coding-agent-integration) · [Completions](#shell-completions) · [Output](#output)

## Core commands

| Command | Purpose | Useful options |
| --- | --- | --- |
| `drove init` | Write the config file with defaults | `-f, --force` overwrites |
| `drove serve` / `drove server` | Start the proxy | `--host`, `--port` |
| `drove server status` | Inspect the running proxy | `-w, --watch [N]`, `-V, --verbose` |
| `drove chat [MODEL]` | Open terminal chat | `-r, --resume`, `-s, --system`, `-e, --endpoint`, `-k, --api-key` |
| `drove config [KEY] [VALUE]` | Show, get, or set global settings | Dotted keys, e.g. `llama_server.n_gpu_layers` |
| `drove models list` | List installed models | `-V, --verbose` |
| `drove models download REF` | Download `org/repo[:QUANT]` | `-n, --name`, `-y, --yes`, `--tts-engine`, `--repair-config`, `-V, --verbose` |
| `drove models add-tts NAME` | Register a TTS engine | `--engine`, `--voice`, `--device` |
| `drove models info NAME` | Inspect a model | `-V, --verbose` |
| `drove models config [NAME] [KEY] [VALUE]` | Show or edit model parameters | `-g, --global`, `--unset KEY`, `--all` |
| `drove models delete NAME` | Remove local model files and config | `-y, --yes` |
| `drove integrate AGENT [MODEL…]` | Configure pi, opencode, or Kilo Code | `-d, --default`, `-n, --dry-run`, `--remove` |
| `drove observe` | Browse request logs in a TUI | `-m, --model` |
| `drove observe web` | Serve the request log web UI | `--host`, `--port` (default `127.0.0.1:8877`) |
| `drove completions generate [SHELL]` | Print a completion script | Shell: bash, zsh, fish, powershell |
| `drove completions install [SHELL]` | Install shell completions | Shell auto-detected if omitted |

Global options: `-c, --config PATH`, `-v, --verbose` (debug logging),
`--version`, and `--help`. For example:

```bash
drove --config /path/to/config.toml server status -V
```

For walkthroughs, see [Managing models](./models.md),
[Text and vision](./text-and-vision.md), and [Running drove](./operations.md).

## Chat model selection

`drove chat` opens a searchable model picker when neither a model argument nor
a configured `default_model` is available.
Type part of a model name to filter, use the arrow keys to move, and press Enter
to choose. Esc or Ctrl+C cancels. A loaded chat model is selected initially when
one is available; otherwise the first model by name is selected. Even a single
model is shown so its details can be inspected before starting.

For drove servers, each row shows loaded state, size on disk, and inferred text,
vision, tools or reasoning capabilities. Speech-to-text and text-to-speech models
are excluded, including backend overrides. Models whose sidecar configuration
cannot be read are not offered. The server's `/v1/models` response retains its
standard fields and every model ID, and adds a `drove` object with `chat`,
`loaded`, `size_bytes`, `capabilities`, and `context_length` for this purpose.

Older drove servers and third-party OpenAI-compatible endpoints may not report
this metadata. Their model entries remain selectable with unknown labels; drove
cannot verify chat capability from a model name alone. Explicit model arguments
remain supported, including `drove chat MODEL --endpoint URL --api-key KEY`.
When stdin or stdout is not a terminal, bypass the picker by passing `MODEL`
or configuring `default_model`.

Inside chat, type `/model` to choose another model without restarting the app.
The current model is preselected. Switching saves the old conversation and starts
a separate conversation for the selected model, preserving the system prompt.
Cancel or select the same model to keep the current conversation. Switching is
unavailable while replies or queued messages are pending. A failed catalog fetch
or session save leaves the current model unchanged; saved conversations remain
available through `/sessions` when using their model.

`drove chat MODEL` uses that model. Without `MODEL`, it uses `default_model`
from configuration, or opens the existing model picker when no default is set.
The same order applies with `--endpoint`; an explicit model overrides the
configured identifier for remote APIs too. `--resume` looks up sessions for
the selected model.

```bash
drove config default_model "org/model:Q4_K_M"
drove chat
drove config default_model ""  # restore picker / server loaded-model fallback
```

See [Default model](configuration.md#default-model) for server behavior and
`DROVE_DEFAULT_MODEL` precedence.

In `drove chat`, user text, thinking, and errors are shown literally: bracketed
text such as `a[i]` is not treated as styling. Assistant replies render Markdown,
while Rich-style tags remain literal. Tool confirmations show the tool
name and exact arguments as plain text. Scroll within the confirmation to review
long arguments; the permission buttons stay available below it.

## Chat controls

Chat keeps the toolbar and composer to one row each, with a single line separating
input from the conversation. Messages use a narrow role marker and minimal padding.
The **Info** button (or `/info`) opens model, endpoint, session, timing and context
details on demand. Token counts appear only when the endpoint reports usage; unknown
counts are labeled **Not reported**. Reply speed is received chunks/s, not tokens/s.
The activity row appears only while replying, running a tool, stopping, or managing
queued messages; while a tool runs it names the tool and its elapsed seconds.

After each reply, a thin bar at the top right of the header shows how full the
model's context is. Hover it for `used / total` tokens; Info shows the same with a
percentage. The bar turns to the warning colour from 80%. When the endpoint does not
report the model's context size (`context_length` on `/v1/models`), the header shows
the raw count instead, such as `ctx 3.2k`; when the reply reported no token usage,
the meter is hidden. It resets on `/new`, `/clear`, a model switch, a session
resume, and a stopped or failed reply, and returns with the next reply.

One compact tip teaches a single action at a time. Sending, inserting a newline,
using commands, opening Info, or stopping a reply retires the corresponding tip.
The **×** button dismisses a tip without changing your draft. Progress is saved in
the active configuration and remembered across conversations and launches; with
no configuration path, it is remembered for the current run. Once the relevant tips
are complete, the guidance row disappears and the conversation gains that space.
Use `/tips` to restart guidance, `/tips off` to hide all tips, and `/help` for the
full command and shortcut reference. Narrow terminals keep **Info** and **+ New**
visible; `/model` and `/sessions` remain available when their buttons are hidden.

In `drove chat`, **Esc stops the current reply** and returns focus to the input.
The partial text and thinking stay visible, are saved to the session, and show a
**Stopped** marker when resumed. Esc while idle does not quit; use **Ctrl+C** or
**Ctrl+Q** to quit. Esc closes slash-command suggestions when idle.

Messages entered during generation appear as **Queued** bubbles. Tab to a queued
message's **Cancel queued message** button to remove it before it is sent. Normal
replies send the queue in order. Stopping a reply or encountering an error pauses
queued messages; **Resume queued** sends them in order. New input joins a paused
queue without restarting it. Pending messages are not saved until sent.

Each tool call appears in the reply as soon as the model requests it, with its
name and shortened arguments, marked **pending** until it is approved and
**running** with elapsed seconds while it executes. `shell_execute` streams the last
few lines of its output into the entry as it runs; the model still receives the
full output. When the tool ends, the same entry shows its result and outcome:
**✓** done, **✗ failed** (an error, timeout, or nonzero exit code), **⊘ denied**,
or **⊘ not run** after a stop. Resumed sessions show earlier calls without an
outcome mark, except denials.

Tool confirmation shortcuts are **y** (allow once), **n** or **Esc** (deny once),
**s** (allow for this session), and **a** (allow always). Esc in this dialog denies
the call rather than stopping the whole turn. A tool that has already started
cannot be undone: stopping shows a waiting status, retains its actual result,
and prevents any remaining calls or follow-up model request. New turns, session
changes, and an explicit quit wait until that tool finishes. Closing the client
stream does not guarantee that a remote backend immediately stops computation.

Use `/save` to save a partial reply without stopping it. `/new`, `/clear`, and
`/sessions` require the current reply (and any running tool) to finish or stop.

## Chat presentation and copying

Assistant replies render Markdown headings, lists and highlighted code blocks while
streaming and when a saved session is resumed. User messages, reasoning, errors and
system notes remain literal text. Links are displayed without opening a browser.

Tab to a message, then press **Ctrl+Y** to copy its original text (including Markdown
source), or **Ctrl+B** to choose a code block with the arrow keys and Enter. The code
picker copies the current block without its fences, preserving code whitespace;
Esc closes the picker. These keys apply only while a message is focused. Return to
the input with Tab to continue typing.

Copying sends an OSC52 clipboard request to the terminal. The terminal must support
and allow clipboard writes; macOS Terminal does not. The confirmation means the
request was sent, not that the operating-system clipboard was verified.

Connection errors suggest `drove server status` and `drove serve`. Proxy errors show
the returned message and an appropriate next step instead of a raw JSON response.
Info's **chunks/s (received)** is client-observed content/reasoning events
per second from the first such event to completion, including tool waits. It is
not a token-generation rate, and events can contain different amounts of text.

`/theme NAME` applies immediately. If saving its configuration fails, a warning
explains that the change applies only to this session; with no configuration path,
the TUI explicitly reports that the theme was not saved.

## Config management

```bash
drove config
drove config idle_timeout_seconds
drove config idle_timeout_seconds 3600
drove config llama_server.n_gpu_layers -1
```

Setting a value writes only that key to the config file; `DROVE_*` overrides
are never saved. It also works when the file holds an invalid value, so it can
repair one; a file with broken TOML syntax is refused unchanged. See [Sources and precedence](configuration.md#sources-and-precedence).

## Models management

```bash
drove models --help
drove models list
drove models download <repo>
drove models delete <name>
drove models info <name>
drove models config <name>
```

The `CAPABILITY` column in `drove models list` shows model capabilities:
`vision` (multimodal projection), `stt` (speech-to-text), and `tts` (text-to-speech).

### Download progress and recovery

Printed retry, resume, and repair inspection commands preserve the effective
`--config` path so recovery uses the same model directory.

Downloads show one overall byte progress display with the active file, transfer
speed and estimated time remaining. Existing complete files are **skipped**;
a transport-reported saved offset is **resumed**. A short final file is marked
**retrying**, because it may need downloading again. Speed counts newly transferred
bytes, excluding existing files and resume offsets. Rate and ETA remain unknown
until transfer samples are available. Redirected output uses bounded plain lines
instead of animated progress bars.

Ctrl+C exits with status 130 and prints a quoted resume command that preserves the
selected quantization, custom name and explicit TTS engine. For an ONNX all-variants
selection, it asks you to choose all variants again (`--yes` selects only the default
ONNX variant). Transfer errors identify
the failed file and exit with status 1. Completed files and Hugging Face's partial
cache are kept: repeat the printed command to skip complete files and reuse cached
partial data where supported. A changed remote file or transport may still require
retransmission. Configuration errors are reported separately from transfer errors.
Their printed retry includes `--repair-config`: it reuses complete payloads, retries
download metadata, and fills missing inferred settings without replacing customized
values. Ordinary already-complete downloads remain read-only. You can inspect the
result with `drove models config <name>`.

The completion report shows the model name, location, actual directory size,
automatic projector/ASR/TTS settings and a next command. Text models suggest
`drove chat <name>`; speech models suggest inspecting their settings and starting
`drove serve` for audio API requests. An already-complete model shows the same useful
summary without downloading again.

For multi-variant downloads, use ↑/↓ and Enter to select a quantization, or
Esc/Ctrl+C to cancel. The picker shows sizes, local status and advisory RAM
estimates. `:QUANT` and `--yes` bypass it for scripts. See
[Choosing a download variant](./models.md#choosing-a-download-variant) for the
memory estimate and download-all behavior.

Before deleting a model, drove lists the files and sidecar configuration it will
remove, their count, and estimated reclaimed space. It checks the configured
server and warns if the model is loaded, including active requests when known.
An unreachable or invalid status response is shown as unknown; offline deletion
still works. This is a snapshot: deletion does not stop a loaded backend, and open
files or hard links can delay or reduce the space reclaimed.

Download confirmation appears directly below the plan summary and states the
approximate amount left to transfer. File details (`--verbose`) and applicable
license notices appear before that summary. Both actions default to **No** when
you press Enter; `--yes` / `-y` skips every question. Deletion with `--yes` also
skips the server-status request.

### Text-to-speech models

```bash
# Download weights; the engine is detected for known repos
drove models download hexgrad/Kokoro-82M --name kokoro
drove models download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --name qwen-tts
drove models download ResembleAI/chatterbox-nano --name chatterbox
drove models download fishaudio/s1-mini --name fish

# A compatible repo that is not auto-detected: name the engine
drove models download <org/repo> --tts-engine kokoro

# Register an engine without downloading; weights go to the Hugging Face cache on first use
drove models add-tts kokoro-british --voice british-male
drove models add-tts qwen-cpu --engine qwen --device cpu
drove models add-tts narrator --engine chatterbox --voice /abs/path/narrator.wav

# Change defaults later
drove models config kokoro tts_voice af_bella
drove models config kokoro tts_device cuda
```

| `models add-tts` option | Meaning |
|--|--|
| `--engine` | `kokoro` (default), `qwen`, `chatterbox`, `fish`, or `kitten`. |
| `--voice` | Default voice, alias, or absolute reference WAV path. Omit to inherit the global setting (`female` otherwise). |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`. Omit to inherit the global setting (`auto` otherwise). |

TTS downloads keep the voice and tokenizer directories and configure the speech
worker automatically. TTS repos have no quantization tags. Voices, formats, and
requests are covered in [Text-to-speech](text-to-speech.md).

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

With no model names, every chat model is added (speech-to-text and
text-to-speech models are skipped) and models drove no longer has are removed from the provider. A
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

## Observe browser

`drove observe` opens the request log browser; `-m MODEL` keeps browsing and
search scoped to one model. Enable logging with `observe = true` in the config
and make requests first. An empty log and a search with no matches show separate
instructions.

The browser loads 100 records at a time. Use **Previous** and **Next** below the
list to change pages, and **r** to refresh from the first page. Press **/** to
focus search; typing waits briefly before searching all record fields. Disk
search runs in the background, so the input remains usable even when a complete
search takes longer. Newer searches replace older results. Full-text search
still scans the log; paging does not create a search index.

While search has focus, **Esc** clears a nonempty query; another **Esc** returns
to the list. **q** quits outside the search input, as does **Ctrl+C**; Esc outside
search retains its quit behavior. Full model names and endpoints are kept in the
list: use **←/→** to scroll long columns horizontally. Select a row to inspect its
cached page snapshot; refresh to reload files changed on disk.

JSON, headers and raw responses follow the active Textual theme, including
`textual-dark` and `textual-light` (configured with `tui_theme`). Unreadable
records are skipped within a page but still counted in an unfiltered total; a
page containing only unreadable records offers navigation or refresh guidance.

## Shell completions

```bash
drove completions --help
```

## Output

![Server status report](../assets/cli-status.svg)

[See the verbose status view](../assets/cli-status-verbose.svg).

Inspection commands use a shared layout with cyan headings, compact tables, and
muted labels. Model names and paths wrap to fit the terminal and are printed
literally, including brackets. Redirected output has no ANSI color codes;
`NO_COLOR=1` disables color in the terminal too.

Human feedback uses explicit `Success:`, `Warning:`, `Error:`, and `Info:` labels,
so outcomes remain distinguishable without color. Warnings and errors go to
standard error; confirmations go to standard output. A `Next:` line gives the
relevant follow-up command. Unknown model names and configuration keys suggest a
close match when available and explain how to list valid choices. Unknown
model-config keys are errors; a valid key with no override still prints `(not set)`.

State-changing commands confirm what they changed after completing their writes.
`drove init` summarizes the saved configuration instead of dumping TOML, and
`drove integrate` shows model limits and capabilities plus written files and
actual backups. Completion installation reports the correct activation step for
the selected shell.

Single-value config queries and generated completion scripts remain plain output
for scripts. Integration and completion `--dry-run` output retains its existing
plain transcript and writes no files; integration previews include model lines,
file-path comments, and JSON rather than a single JSON document.

Default reports prioritize the next decision; detail is available on demand:

| Command | Default view | More detail |
| --- | --- | --- |
| `server status` | Endpoint, uptime, active requests, errors, network exposure (non-loopback bind only), loaded models, approximate idle-unload countdown | `--verbose` / `-V`: listen address, load/idle durations, backend memory/CPU, token counts, speed, first-token latency |
| `models list` | Name, capability, size, total disk usage | `--verbose` / `-V`: storage directory, custom-config indicator, download origin/files |
| `models info <name>` | Capabilities, size/file count, origin, quantization, explicit model overrides | `--verbose` / `-V`: local paths and downloaded file inventory |
| `models config <name>` | Effective non-null values and their sources | `--all`: include unset parameters |
| `models download <repo>` | Model, destination, file count, total size, approximate remaining transfer, completed-file count | `--verbose` / `-V`: every file and its resume state |
| `models delete <name>` | Exact files and sidecars, estimated space, loaded-state warning, safe confirmation | `--yes` / `-y`: skip confirmation and status request |

These options follow the subcommand (for example, `drove server status -V`).

Audio traffic in `drove server status` includes successful transcription and speech
request counts, audio seconds, and processing seconds since startup. Processing time
includes the backend request and response transfer, excluding model startup. Audio
duration is reported by the bundled workers for all supported response formats.
The root `drove --verbose` option still controls debug logging.

Status always shows nonzero errors, labeled as cumulative since server startup.
The endpoint URL is the address `status` connects to, so a `0.0.0.0` bind shows as
`127.0.0.1`. When the server listens on anything other than loopback — the default
`listen_host = "0.0.0.0"` included — a `listening on …` line shows the real bind
address, because the server is reachable from other machines.
An idle-unload countdown is approximate: active requests prevent idle unloading,
and eviction or configuration changes can unload a model sooner. Busy models show
`—`; expired timers show `due`. Download estimates use remote file sizes and local
partial files; the actual transfer may vary with cache reuse or retries.

`--watch` refreshes the screen only in a terminal; redirected watch output appends
snapshots. Stop watching with Ctrl+C. An unused server stays concise:

```text
drove / Server
http://127.0.0.1:8080/v1

running · up 12m 4s · 0 active requests
listening on all interfaces (0.0.0.0:8080)

Loaded models · 0
No models loaded. Models start on the first request.
Use --verbose for process, timing, and token details.
```

Configuration reports retain explicit `0` and `false` values. Unset model
parameters are hidden with a count and a hint to use `--all`. Global
`drove config` continues to show all application settings, and single-value
queries such as `drove config listen_port` remain plain values for shell scripts.
Confirmation prompts share the report style, include the action consequences,
and default to No. Declining or reaching end-of-input aborts with a nonzero exit code.

[Docs](./README.md) · [CLI](./cli.md) · [Configuration](./configuration.md) · [Troubleshooting](./troubleshooting.md)
