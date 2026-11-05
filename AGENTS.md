# drove — Agent guide

Guidance for AI agents (and humans) contributing to **drove**.

`drove` is a llama.cpp server manager and proxy — like ollama, but wrapping
`llama-server` directly. It lazily starts a backend on the first request for a
model (`.gguf` → `llama-server`, `.onnx` speech-to-text → a built-in ASR
worker, `.tts` text-to-speech → a bundled TTS worker), proxies OpenAI-format API traffic to it, evicts models LRU-first when
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
  m4a, ogg…) and the TTS worker return mp3/opus/aac/flac or change speed;
  without it, WAV only.
- **espeak-ng** (and **sox** for Qwen3-TTS) — only to *serve* text-to-speech;
  the TTS worker also needs `uv` on `PATH` to build its Python 3.12 engine
  environments. Not required to run the tests.
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
- **Coverage (report only):** `make cov` (line coverage with missing lines)
- **Release rehearsal:** `make release-check` — two pinned PSR releases and
  original-commit publication retries in a disposable repository; no remote writes.
- **Test (single file / focused):** no make target exists — the one sanctioned
  raw invocation:

  ```bash
  uv run pytest tests/test_proxy.py
  uv run pytest tests/test_proxy.py::test_name
  ```

- **Run locally:** `drove serve` after `make install` (equivalently
  `drove server`; `serve` is a hidden alias), then point any OpenAI client at
  `http://localhost:8080/v1`. Useful stablemates: `drove chat`,
  `drove models list`, `drove server status`, `drove observe`,
  `drove integrate <agent>` (pi, opencode, or Kilo Code).
- **Completions:** `make completions` (runs `make install` first).
- **Systemd user service:** `make service-install` (writes
  `~/.config/systemd/user/drove.service`, enables + starts it), plus
  `make service-start`, `service-stop`, `service-restart`, `service-status`,
  `service-logs`, `service-uninstall`.

Always run `make lint`, `make typecheck`, and `make test` before opening a PR.
The **Check** workflow re-runs them (plus `uv lock --check`) on every PR and
before every release, but a red check costs a round trip (see
[Gotchas](#gotchas)).

## Golden rules

1. **Start every task in a separate Git worktree under `/tmp`.** Create a
   new branch from the latest `origin/main` and do all edits and checks in
   that worktree. Leave the original checkout and other worktrees untouched.
   **Never commit directly to `main`.** Always branch, always PR — and
   squash-merge. Merging can cut a release, and the squash commit (which
   carries the PR title) is what the release automation parses.
2. **Never force-push a shared branch.**
3. **Keep `main` green.** The **Check** workflow runs `uv lock --check`,
   `make lint`, `make typecheck`, and `make test` on every PR, and
   semantic-release waits for the same check on `main` — a red `main` never
   releases. Run them yourself before opening a PR (see
   [Run the checks locally](#5-run-the-checks-locally)).
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
8. **Drove configuration flows through `src/drove/config.py`.** `DROVE_*`
   settings and the global TOML file are read only there. Reading another
   tool's documented environment variables in the module that integrates
   with that tool is allowed: `SHELL` in `cli/completions.py`, and
   `XDG_CONFIG_HOME`, agent config-path variables (`OPENCODE_CONFIG`,
   `KILO_CONFIG`) and `PI_CODING_AGENT_DIR` in `integrations.py`. Per-model
   settings belong in sidecar TOMLs parsed by `model_config.py`, never hardcoded.
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

### 1. Fetch the latest `main`

From the repository checkout, fetch without switching branches or changing
any existing work:

```bash
git fetch origin main
```

### 2. Create a branch and worktree under `/tmp`

Start every task in its own worktree. Use a unique directory under `/tmp`
and a new branch based on `origin/main`:

```bash
worktree_dir=$(mktemp -d /tmp/drove-XXXXXXXX)
git worktree add -b docs/short-description "$worktree_dir" origin/main
cd "$worktree_dir"
```

Replace `docs/short-description` with the branch name for your task. Run all
edits, checks, commits, and PR commands from this worktree; do not switch
branches or modify files in the original checkout or another task's worktree.

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
`build`, `ci`, `chore`. Add `!` before the colon for a breaking
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

CI runs these too (the **Check** workflow), but do not open a PR with them
failing:

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

- Make sure the **PR Title** check, the **Check** workflow (`uv lock
  --check`, lint, type-check, tests), and the **Install** workflow (a
  `make install` + CLI smoke test) are green.
- Address review feedback by pushing more commits to the same branch.
- Mention `@claude` in a PR/issue comment to have Claude Code act on it
  (`.github/workflows/claude.yml`).
- Maintainers squash-merge: verify the prefilled commit title is the PR title
  and the body is the PR description, preserving intentional breaking footers (see [Gotchas](#gotchas)).

## Release automation

- **Is `main` protected?** Direct pushes to `main` are forbidden by policy
  ([docs/deploy.md](docs/deploy.md)); everything lands via squash-merged PRs,
  which is what makes the PR title become the commit message the automation
  parses.
- **What does merging trigger?** `.github/workflows/release.yml` runs
  [python-semantic-release](https://python-semantic-release.readthedocs.io/) 9.21.2
  on every push to `main`, after the **Check** workflow passes on that
  commit (`needs: check`). When a release is due it bumps `project.version` in
  `pyproject.toml`, runs `scripts/release.py prepare` to promote reviewed `## [Unreleased]`
  bullets into `## vX.Y.Z (YYYY-MM-DD)` and `uv lock` to refresh the lock,
  commits `chore(release): vX.Y.Z` with `[skip ci]`, tags
  `v{version}`, and the workflow publishes that tag's handwritten changelog section with
  `scripts/release.py publish` (safe to retry after publication failure).
  PSR runs with `--no-changelog --no-vcs-release`; its build assets include
  `CHANGELOG.md` and `uv.lock`, so versions and lock are committed together.
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
                       + DROVE_* env overrides — the only reader of drove settings
  model_config.py      Per-model sidecar TOMLs → llama-server args (ctx_size,
                       n_gpu_layers, cache_*, extra_args passthrough, backend,
                       asr_model, asr_quantization, tts_*)
  model_store.py       The single authority for resolving/listing models on disk;
                       lists and serves only complete downloads
  download_state.py    Per-model .drove-download.json record (incomplete/complete,
                       expected files and sizes)
  atomic_write.py      The one atomic file writer: exclusive temp file beside the
                       target, rename into place, cleanup on failure
  backend.py           Picks the backend per model: .gguf → llama-server,
                       .onnx → built-in ASR worker, .tts → TTS worker
                       (overridable via `backend`); known TTS repos → engine
  server_manager.py    Subprocess lifecycle (asyncio.subprocess): lazy start,
                       /health wait, idle shutdown, LRU-first + memory-budget
                       eviction, prompt-cache save/restore across sleep-wake
  workers/asr.py       Built-in ONNX speech-to-text worker (python -m
                       drove.workers.asr); onnx-asr comes with the optional
                       `drove[asr]` extra
  workers/tts_runtime/ Text-to-speech worker (worker.py) and the Kokoro, Qwen3-TTS,
                       Chatterbox and Fish engines adapted from Cathy (CATHY-LICENSE),
                       plus Kitten (engines.py; model terms in KITTEN-NOTICE);
                       run by `uv run --python 3.12` with the
                       per-engine <engine>.txt requirements, never imported by drove
  proxy.py             FastAPI reverse proxy: extracts the model from JSON or
                       multipart bodies, ensures the backend is running,
                       forwards via httpx, resets the idle timer, records
                       observe logs when enabled
  downloader.py        HuggingFace downloads (GGUF + ONNX + TTS repos, quantization
                       prompts)
  integrations.py      Registers models with pi, opencode, and Kilo Code; merges
                       a drove provider into each agent's JSON config, inferring
                       capabilities from model config and GGUF metadata
  gguf.py              Minimal GGUF header reader (sliding-window detection)
  observe.py           Request/response record storage
  observe_tui.py       Textual TUI browser for observe logs
  observe_web.py (+ observe_web_static/)   Small web UI for observe logs
  chat_models.py       Chat catalog metadata and OpenAI-compatible model-list clients
  model_picker.py      Searchable model picker shared by startup and chat
  sessions.py, tui.py, tools.py, stats.py Chat TUI app, session persistence,
                       tool execution, token/process metrics
  tool_policy.py       Chat tool approvals (cwd/remote reads, scoped grants,
                       exact-command shell grants, per-call writes) and the 64 KiB tool output cap,
                       testable without the TUI
  cli/                 Typer CLI: main.py (config, chat, observe, init),
                       models.py (list, download, add-tts, delete, info, config),
                       server.py (serve — default, status), integrate.py
                       (agent registration), completions.py, tui_logging.py
                       (logging rerouted while a Textual app runs), output.py
                       (shared CLI presentation), download_progress.py
                       (terminal and redirected progress), quant_picker.py
                       (quantization previews and interactive picker)
scripts/release.py      Handwritten release notes promotion and retryable publication
scripts/check_release.py  Disposable two-release rehearsal (`make release-check`)
tests/                 pytest module and feature suites — see tests/
docs/                  User and contributor guides — index in docs/README.md;
                       proposals/ holds design proposals
.github/workflows/     check.yml (lock check, lint, typecheck, test), install.yml
                       (make install smoke), pr-title.yml, release.yml (check,
                       then semantic-release), claude.yml (@claude)
```

Layering: `proxy.py` receives requests and delegates lifecycle to
`server_manager.py`, which picks the command via `backend.py` and resolves
paths via `model_store.py`; drove settings come only from `config.py`,
per-model overrides only from `model_config.py`. Integration modules may
read the third-party environment variables described in golden rule 8.
The CLI and TUI are thin fronts over the same pieces.

Request flow:

```
Client → FastAPI proxy (drove port)
           → ServerManager.ensure_running(model)
               → if not running: spawn backend subprocess
                   (GGUF → llama-server, ONNX speech-to-text → python -m drove.workers.asr,
                    .tts text-to-speech → uv run --python 3.12 … tts_runtime/worker.py)
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
- **Configuration:** `DROVE_*` settings and the global TOML file are read only
  in `config.py`; per-model settings only via sidecar TOMLs (`drove models
  config` edits them). Integration modules may read another tool's documented
  environment variables (see golden rule 8).
- **Interface design:** prioritize space for active content. Use thin delimiters
  and minimal padding; show diagnostics on demand. Teach one action at a time in
  a single compact guidance area, retire tips after use or dismissal, remember
  progress across launches, and let users reopen help. Keep actionable states
  (errors, stopping, paused queues, permission choices) visible when relevant.
- **Async:** the server path is fully async; tests run with
  `asyncio_mode = "auto"` — no markers needed.
- **Registering new components:** a new llama-server flag becomes a
  `ModelConfig` field (mapped to its CLI arg); a new CLI command is a Typer
  command in `src/drove/cli/`; new config keys are `Config` fields listed in
  `write_default_config()`, and are saved with `set_config_value()` (one key,
  never the env-loaded config).

## Testing

Coverage is report-only: run `make cov` to identify gaps; no global percentage
floor is enforced. New modules should ship with tests, including TUI modules
(using Textual pilots where appropriate). Test observable behavior and regression
- **Location & naming:** `tests/test_<module>.py`, one per source module —
  `test_proxy.py`, `test_server_manager.py` (lifecycle, eviction races,
  health checks), `test_config.py`, `test_model_config.py`,
  `test_model_store.py`, `test_backend.py`, `test_asr_worker.py`,
  `test_downloader.py`, `test_gguf.py`, `test_prompt_cache.py`,
  `test_observe.py`, `test_observe_web.py`, `test_proxy_observe.py`,
  `test_cli_models.py`, and the text-to-speech suites `test_tts_worker.py`,
  `test_tts_integration.py`, `test_tts_download.py`, `test_release.py` (reviewed
  notes promotion and retryable publication), plus `test_workflows.py`
  (the CI check and release-gate contract in `.github/workflows/`).
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
- Model names and config tokens (`backend`, `asr_model`, `asr_quantization`,
  `tts_*`) end up on subprocess command lines; they are validated against
  allowlists (`VALID_BACKENDS`, `Literal` fields such as `tts_engine` and
  `tts_device`, safe-token shape checks) — keep new ones validated the same way.
- TTS reference voices are fixed by trusted local config; cloning engines
  reject any request voice other than the configured default (`alloy`), so API
  callers can never point the server at an arbitrary file. Don't regress this.
- ASR uploads are capped at 100 MB (HTTP 413 past it) and ffmpeg stderr is
  never echoed back to clients (generic error instead). Don't regress these.

## Gotchas

- **CI checks the lockfile before anything else.** `uv run` (behind every
  make target) silently rewrites a stale `uv.lock`, so the Check workflow runs
  `uv lock --check` first. If it fails, run `uv lock` and commit the result.
- **A never-awaited coroutine fails the suite.** pytest turns `RuntimeWarning`
  and unraisable-exception warnings into errors. Mock an async iterator such
  as `aiter_raw` with a plain callable (`lambda: aiter([...])`), not
  `AsyncMock`, which returns a coroutine.
- **Undeclared sidecar TOML keys are dropped silently** (`extra="ignore"`).
  If a flag seems ignored, check the key is a real `ModelConfig` field — or
  use `extra_args`. The context-size field is `ctx_size`, not `context_size`.
- **`n_parallel` maps to `--parallel`**, not `--n-parallel` — llama-server
  rejects the latter outright and the model never starts.
- **Squash-merge with the PR title as the commit title and PR description as the body.** If GitHub
  prefills the branch's commit list instead, semantic-release parses the
  wrong messages and the bump is wrong. Check the merge dialog and retain intentional breaking footers. Quote literal
  `BREAKING CHANGE:` examples with backticks so they cannot trigger a bump.
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
