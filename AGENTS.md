# drove — Agent guide

Guidance for AI agents (and humans) contributing to **drove**.

`drove` is a llama.cpp server manager and proxy — like ollama, but wrapping
`llama-server` directly. It lazily starts a backend on the first request for a
model (`.gguf` → `llama-server`, `.onnx` speech-to-text → a built-in ASR
worker), proxies OpenAI-format API traffic to it, evicts models LRU-first when
the model-count or memory budget is hit, and shuts each one down after a
configurable idle timeout. A change is "done" when `make lint`, `make
typecheck`, and `make test` pass and `CHANGELOG.md` carries an entry for it.

This project follows GitHub flow: `main` is always releasable, all work happens
on short-lived branches, and every change lands through a squash-merged pull
request. Merging to `main` can cut a release — the PR title decides whether and
how big (see [Release automation](#release-automation)).

Read [README.md](README.md) for setup and [docs/architecture.md](docs/architecture.md)
for the full design. This file is the operational checklist for *how to work*
here.

## Prerequisites

- **Python ≥ 3.14** — `pyproject.toml` requires it; the Makefile pins
  `DROVE_PYTHON ?= 3.14`. uv downloads the interpreter, so it doesn't need to
  be preinstalled.
- **[uv](https://docs.astral.sh/uv/)** — the only toolchain. `make install`
  bootstraps it if missing.
- **`llama-server`** from [llama.cpp](https://github.com/ggml-org/llama.cpp) on
  `PATH` — required to *serve* GGUF models; **not** required to run the tests
  (they fake the subprocesses).
- **ffmpeg** (optional) — lets the ASR worker accept compressed audio (mp3,
  m4a, ogg…); without it, WAV only.
- **`gh` CLI** — for pull requests.
- **Linux or macOS** — `make install` refuses other OSes.

## Commands

`make` owns the dev flow. **Always use the make targets rather than invoking
`uv`/`pytest`/`ruff`/`mypy` directly.**

- **Install (global CLI):** `make install` — installs `drove` as a uv tool,
  with the `asr` extra by default (`make install DROVE_EXTRAS=` for a
  text-only install). Not a prerequisite for development: the targets below use
  `uv run`, which syncs the project environment automatically.
- **Lint:** `make lint` (`ruff check .`)
- **Format:** `make fmt` (`ruff format .`)
- **Type-check:** `make typecheck` (`mypy src/`, strict)
- **Test (all):** `make test` (`pytest`)
- **Test (single file / focused):** no make target exists — the one sanctioned
  raw invocation:

  ```bash
  uv run pytest tests/test_proxy.py
  uv run pytest tests/test_proxy.py::test_name
  ```

- **Run locally:** `drove serve` after `make install` (equivalently
  `drove server`; `serve` is a hidden alias), then point any OpenAI client at
  `http://localhost:8080/v1`. Useful stablemates: `drove chat`,
  `drove models list`, `drove server status`, `drove observe`.
- **Completions:** `make completions` (runs `make install` first).
- **Systemd user service:** `make service-install` (writes
  `~/.config/systemd/user/drove.service`, enables + starts it), plus
  `make service-start`, `service-stop`, `service-restart`, `service-status`,
  `service-logs`, `service-uninstall`.

Always run `make lint` and `make test` before opening a PR — CI does **not**
run them (see [Gotchas](#gotchas)).

## Golden rules

1. **Never commit directly to `main`.** Always branch, always PR — and
   squash-merge. Merging can cut a release, and the squash commit (which
   carries the PR title) is what the release automation parses.
2. **Never force-push a shared branch.**
3. **Keep `main` green.** CI only smoke-tests `make install` and lints PR
   titles — `make lint`, `make typecheck`, and `make test` are *your* job
   before opening a PR (see [Run the checks locally](#5-run-the-checks-locally)).
4. **Use the project's task runner.** `make` is the single source of the dev
   flow — don't hand-roll the underlying `uv`/`pytest`/`ruff`/`mypy` commands
   (single-test `uv run pytest` is the one fallback, because no target exists
   for it). If the flow needs to change, change the Makefile so everyone
   stays in sync.
5. **Never disable, skip, or delete a test to make a build pass.** If a test
   is wrong, say so and propose the fix.
6. **The PR title is load-bearing.** It must be a valid Conventional Commits
   string (a CI check blocks the merge otherwise), and it alone decides
   whether merging cuts a release and how big the bump is
   (see [PR titles](#pr-titles)).
7. **Every change gets a `CHANGELOG.md` entry under `## [Unreleased]` before
   it is committed** — bug fixes, features, refactors, doc updates, and
   dependency bumps; no exceptions. Keep-a-Changelog groups (`### Added`,
   `### Changed`, `### Deprecated`, `### Removed`, `### Fixed`,
   `### Security` — create the subsection if it doesn't exist), one bullet per
   user-visible change, past tense, impact rather than implementation, never
   a version number or date. The entry must be consistent with the commit
   type. The release automation promotes `[Unreleased]` into a versioned
   section when it cuts a release.
8. **All configuration flows through `src/drove/config.py`.** It is the only
   place the environment (`DROVE_*`) and the TOML file are read — never read
   env vars ad hoc in business code. Per-model settings belong in sidecar
   TOMLs parsed by `model_config.py`, never hardcoded.
9. **Never manually edit `project.version` in `pyproject.toml`, create
   `vX.Y.Z` tags or GitHub Releases, or add version headings to
   `CHANGELOG.md`.** python-semantic-release owns all of it; hand edits
   desynchronize it.

## Communication

- Always explain the reasoning behind decisions and approaches.
- When claiming something works or is fixed, prove it — a passing test, a
  script that validates the behavior, or a clear explanation of why. Don't just
  assert.
- When uncertain, say so rather than presenting a guess as fact.
- End each response with a confidence indicator: 🟢 High | 🟡 Medium | 🔴 Low

## The GitHub flow, step by step

### 1. Start from an up-to-date `main`

```bash
git checkout main
git pull origin main
```

### 2. Create a branch

Branch names are short, lowercase, hyphenated, and prefixed by intent, matching
the Conventional Commits type you expect the PR to use:

```
feat/<short-description>      # new feature
fix/<short-description>       # bug fix
refactor/<short-description>  # internal change, no behavior change
perf/<short-description>      # performance work
docs/<short-description>      # documentation only
chore/<short-description>     # tooling, deps, housekeeping
```

Examples: `fix/proxy-503-unreachable`, `feat/prompt-cache`.

### 3. Make focused changes

- One logical change per PR — and a whole feature *is* one logical change.
  Ship its code, tests, and docs together; don't split it across a chain of
  dependent PRs. Don't bundle an unrelated refactor into a fix either.
- Match the surrounding style: async throughout, typed pydantic models for
  config, Typer command modules under `src/drove/cli/`, early returns.
- Keep diffs focused: everything in the diff should serve that one change —
  including the `CHANGELOG.md` `[Unreleased]` bullet (golden rule 7), which
  belongs in the same PR.

### 4. Commit

Commits follow [Conventional Commits](https://www.conventionalcommits.org):

```
type(optional-scope): short imperative description
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore`, `revert`. Add `!` before the colon for a breaking
change.

```
fix(proxy): return 503 when llama-server is unreachable
feat(server): keep the prompt cache across model sleep and wake
feat(api)!: rename the completions request schema
```

Write in the imperative mood ("add", not "added"). Keep the subject under ~72
characters and explain the *why* in the body when it isn't obvious. On merge
only the squash commit — the PR title — is parsed, but well-formed branch
commits keep reviews honest.

### 5. Run the checks locally

CI will not run these — do not open a PR with them failing:

```bash
make lint
make test
make fmt        # before lint, if you touched code it formats
make typecheck  # when you touched typed code; strict over src/
```

### 6. Push and open a PR

```bash
git push -u origin docs/<short-description>
gh pr create --base main --fill
```

Target **`main`**.

## PR titles

The PR title follows the same Conventional Commits format as commits:

```
type(optional scope)!: description
```

A CI check (`.github/workflows/pr-title.yml`, running
`amannn/action-semantic-pull-request@v6`) blocks the merge on an invalid
title: the type must be one of `build`, `chore`, `ci`, `docs`, `feat`, `fix`,
`perf`, `refactor`, `style`, `test`; a scope is optional; the subject must
start with a letter and not end with a period. Editing the title re-validates
it.

The title also decides the release bump — see [Release
automation](#release-automation).

## PR description

Keep it short and useful:

- **What** changed and **why** (the motivation/problem).
- **How to test** / what you ran (`make lint`, `make test`, any manual check).
- **Linked issues**: `Closes #123` when it resolves one.
- For breaking changes: a `BREAKING CHANGE: …` footer in the description body —
  it lands in the squash commit and forces the major bump.

## After opening the PR

- Make sure the **PR Title** check is green and the **Install** workflow (a
  `make install` + CLI smoke test) passes. These are the only CI jobs —
  lint/test/typecheck ran locally or not at all.
- Address review feedback by pushing more commits to the same branch.
- Mention `@claude` in a PR/issue comment to have Claude Code act on it
  (`.github/workflows/claude.yml`).
- Maintainers squash-merge: verify the prefilled commit message is the PR title
  alone, not the branch's commit list (see [Gotchas](#gotchas)).

## Release automation

- **Is `main` protected?** Direct pushes to `main` are forbidden by policy
  ([docs/deploy.md](docs/deploy.md)); everything lands via squash-merged PRs,
  which is what makes the PR title become the commit message the automation
  parses.
- **What does merging trigger?** `.github/workflows/release.yml` runs
  [python-semantic-release](https://python-semantic-release.readthedocs.io/) v9
  on every push to `main`. When a release is due it bumps `project.version` in
  `pyproject.toml`, rewrites `CHANGELOG.md` (promoting `## [Unreleased]` into a
  versioned section), commits `chore(release): vX.Y.Z` with `[skip ci]`, tags
  `v{version}`, and publishes a GitHub Release with the changelog as the body.
  There is no PyPI publish — drove is distributed via `uv tool install` from
  this GitHub repository, smoke-tested by the Install workflow.
- **Does the PR title decide the bump?** Yes. The squash-merge commit carries
  the PR title, and that commit is what semantic-release parses
  (`minor_tags = ["feat"]`, `patch_tags = ["fix", "perf"]` in
  `pyproject.toml`):

| PR title prefix | Release effect |
| --- | --- |
| `feat!: …` (any `type!:`), or a `BREAKING CHANGE:` footer | **major** |
| `feat: …` | **minor** |
| `fix: …`, `perf: …` | **patch** |
| `docs:`, `refactor:`, `style:`, `test:`, `build:`, `ci:`, `chore:` | **none** |

> ⚠️ Choose the prefix deliberately — it decides whether (and how big) a
> release ships when the PR merges. A `refactor:` PR never releases.

## Project map (where things live)

```
src/drove/
  config.py            Global config (pydantic-settings): ~/.config/drove/config.toml
                       + DROVE_* env overrides — the ONLY place env is read
  model_config.py      Per-model sidecar TOMLs → llama-server args (ctx_size,
                       n_gpu_layers, cache_*, extra_args passthrough, backend,
                       asr_model, asr_quantization)
  model_store.py       The single authority for resolving/listing models on disk
  backend.py           Picks the backend per model: .gguf → llama-server,
                       .onnx → built-in ASR worker (overridable via `backend`)
  server_manager.py    Subprocess lifecycle (asyncio.subprocess): lazy start,
                       /health wait, idle shutdown, LRU-first + memory-budget
                       eviction, prompt-cache save/restore across sleep-wake
  workers/asr.py       Built-in ONNX speech-to-text worker (python -m
                       drove.workers.asr); onnx-asr comes with the optional
                       `drove[asr]` extra
  proxy.py             FastAPI reverse proxy: extracts the model from JSON or
                       multipart bodies, ensures the backend is running,
                       forwards via httpx, resets the idle timer, records
                       observe logs when enabled
  downloader.py        HuggingFace downloads (GGUF + ONNX, quantization prompts)
  converter.py         HF → GGUF conversion
  gguf.py              Minimal GGUF header reader (sliding-window detection)
  observe.py           Request/response record storage
  observe_tui.py       Textual TUI browser for observe logs
  observe_web.py (+ observe_web_static/)   Small web UI for observe logs
  sessions.py, tui.py, tools.py, stats.py Chat TUI app, session persistence,
                       tool execution, token/process metrics
  cli/                 Typer CLI: main.py (config, chat, observe, init),
                       models.py (list, download, delete, info, config),
                       server.py (serve — default, status), completions.py
tests/                 pytest suite, one file per module
docs/                  architecture.md, configuration.md, deploy.md (release
                       process), getting-started.md, cli.md, speech-to-text.md
.github/workflows/     install.yml (make install smoke), pr-title.yml,
                       release.yml (semantic-release), claude.yml (@claude)
```

Layering: `proxy.py` receives requests and delegates lifecycle to
`server_manager.py`, which picks the command via `backend.py` and resolves
paths via `model_store.py`; settings come only from `config.py`, per-model
overrides only from `model_config.py`. The CLI and TUI are thin fronts over
the same pieces.

Request flow:

```
Client → FastAPI proxy (drove port)
           → ServerManager.ensure_running(model)
               → if not running: spawn backend subprocess
                   (GGUF → llama-server, ONNX speech-to-text → python -m drove.workers.asr)
               → wait for backend /health
           → httpx reverse proxy → backend port
           → reset inactivity timer
```

Models live in a flat directory (default `~/.local/share/drove/models/`); the
filename without extension is the model name, and each model may carry a
sidecar `<model_name>.toml`.

Config file format (source priority: `DROVE_*` env > TOML > field defaults):

```toml
# ~/.config/drove/config.toml (or $DROVE_CONFIG)
models_dir = "~/.local/share/drove/models"
listen_host = "0.0.0.0"
listen_port = 8080
llama_server_bin = "llama-server"
idle_timeout_seconds = 1800        # default 30 min
max_loaded_models = 1              # LRU eviction at capacity
prompt_cache = true                # KV cache saved/restored across idle shutdowns

[llama_server]
# default llama-server args applied to all models; overridable per sidecar
n_gpu_layers = -1
```

Per-model config format — every key must be a declared `ModelConfig` field or
it is silently ignored; `extra_args` is appended verbatim after every
generated argument (llama-server takes the last occurrence, so these win):

```toml
# ~/.local/share/drove/models/<name>.toml
ctx_size = 4096
n_gpu_layers = -1
extra_args = ["--a-flag-llama-server-added-yesterday", "value"]
```

## Conventions

- **Style:** ruff, line length 100, `target-version = "py314"`, rules
  `E`, `F`, `I`, `UP`; `make fmt` formats, `make lint` checks.
- **Typing:** mypy strict over `src/` (`make typecheck`). `onnx_asr` is the
  only import exemption (optional extra) — don't widen it.
- **Configuration:** global settings only via `config.py`; per-model settings
  only via sidecar TOMLs (`drove models config` edits them). No scattered env
  reads.
- **Async:** the server path is fully async; tests run with
  `asyncio_mode = "auto"` — no markers needed.
- **Registering new components:** a new llama-server flag becomes a
  `ModelConfig` field (mapped to its CLI arg); a new CLI command is a Typer
  command in `src/drove/cli/`; new config keys go through `Config` and its
  `save()`.

## Testing

- **Framework / runner:** pytest (`make test`), asyncio auto mode,
  `testpaths = ["tests"]`.
- **Location & naming:** `tests/test_<module>.py`, one per source module —
  `test_proxy.py`, `test_server_manager.py` (lifecycle, eviction races,
  health checks), `test_config.py`, `test_model_config.py`,
  `test_model_store.py`, `test_backend.py`, `test_asr_worker.py`,
  `test_downloader.py`, `test_gguf.py`, `test_prompt_cache.py`,
  `test_observe.py`, `test_observe_web.py`, `test_proxy_observe.py`,
  `test_cli_models.py`.
- **What to cover:** happy path + error paths; lifecycle and eviction changes
  ship with regression tests for the race they fix (the existing suites set
  the bar).
- **Fixtures / stubs:** tests fake the backends (patched start/health, temp
  model dirs) — no `llama-server`, GPU, network, or downloaded models needed;
  the suite runs anywhere uv does.

## Security

- Never commit secrets, API keys, credentials, or sensitive data. drove's
  config holds local paths and ports only — there are no API keys by design;
  keep it that way.
- The proxy listens on `0.0.0.0` by default — it is LAN-exposed. Treat every
  proxied payload as external input.
- Model names and config tokens (`backend`, `asr_model`, `asr_quantization`)
  end up on subprocess command lines; they are validated against allowlists
  (`VALID_BACKENDS`, safe-token shape checks) — keep new ones validated the
  same way.
- ASR uploads are capped at 100 MB (HTTP 413 past it) and ffmpeg stderr is
  never echoed back to clients (generic error instead). Don't regress these.

## Gotchas

- **CI never runs `make test`/`lint`/`typecheck`.** The only PR checks are
  the title lint and the `make install` smoke test — the local gate is the
  only gate.
- **Undeclared sidecar TOML keys are dropped silently** (`extra="ignore"`).
  If a flag seems ignored, check the key is a real `ModelConfig` field — or
  use `extra_args`. The context-size field is `ctx_size`, not `context_size`.
- **`n_parallel` maps to `--parallel`**, not `--n-parallel` — llama-server
  rejects the latter outright and the model never starts.
- **Squash-merge with the PR title as the commit message.** If GitHub
  prefills the branch's commit list instead, semantic-release parses the
  wrong messages and the bump is wrong. Check the merge dialog.
- **`make install` is not dev setup.** It installs a *global* uv tool
  (Python 3.14, `asr` extra by default); the dev targets work without it.
- **Version, tags, releases, and versioned changelog sections are
  automation-owned** — docs/deploy.md lists what never to touch by hand.
- **`drove serve` and `drove server` are the same command** (`serve` is a
  hidden alias of the `server` group's default action); `drove server status`
  is the visible subcommand.

## Start here

The fastest path to understanding this codebase:

[docs/architecture.md](docs/architecture.md) → [src/drove/proxy.py](src/drove/proxy.py) →
[src/drove/server_manager.py](src/drove/server_manager.py) → [src/drove/config.py](src/drove/config.py)
