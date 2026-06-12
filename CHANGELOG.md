# CHANGELOG

## [Unreleased]

### Added

- Speech-to-text models (e.g. NVIDIA Parakeet TDT) served through a new built-in
  ASR worker (`drove.workers.asr`) with an OpenAI-compatible
  `/v1/audio/transcriptions` endpoint, managed with the same lazy-start /
  idle-shutdown lifecycle as llama-server. Requires the optional `drove[asr]`
  extra (onnx-asr).
- ONNX model support across model management: `drove models download` resolves
  ONNX repos (including int8 variants via `org/repo:int8` and support files such
  as `vocab.txt`), and `models list`/`info`/`delete`/`config` handle ASR models
  like any other, with a new `stt` capability tag.
- Per-model `backend`, `asr_model`, and `asr_quantization` config keys; the
  backend is auto-detected from the model files (`.onnx` → ASR worker) and the
  ASR model type is auto-configured at download time for known repos.
- The proxy now extracts the `model` field from multipart form bodies, so
  OpenAI-style audio requests route to the right model on the same listen port.
- Expanded ASR test coverage: ffmpeg decode-failure and conversion happy
  paths, non-16-bit and corrupt WAV handling, multi-channel downmix, the
  worker CLI entry point, and the ASR model-type inference fallback in
  `server_manager`.

### Changed

- Clarified the quantization-filtering logic in the downloader
  (`filter_onnx_quant`) by checking the explicit-quant case first; no
  behavior change.
- Updated `docs/architecture.md` to describe the backend-per-model design
  (llama-server for GGUF, the built-in ASR worker for ONNX speech-to-text)
  instead of llama-server only.
- The install script and `make install` now include the `asr` extra by
  default, so speech-to-text models work out of the box; set `DROVE_EXTRAS=""`
  on the install script for a minimal text-generation-only install.
- Reworked the README around the two model classes drove serves: text
  generation (Gemma example via llama-server) and speech-to-text (Parakeet
  example via the built-in ONNX worker), with curl examples for both.


## v0.1.2 (2026-06-12)

### Bug Fixes

- Version flag, llama-server startup warning, TUI tool-grant reset
  ([`e1aa1eb`](https://github.com/cleanunicorn/drove/commit/e1aa1eb27d76edb6c9ee156ee2b1bb29c7d84839))

- Add a --version flag to the drove CLI (install.sh already probes `drove --version` during
  installation). - Warn at `drove server` startup when the configured llama_server_bin is not found
  on PATH, instead of failing only on the first request. - Reset session-level tool permissions when
  starting a new chat via /new or /clear in the chat TUI, so grants no longer leak across
  conversations.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>


## v0.1.1 (2026-05-20)


## v0.1.0 (2026-05-20)

### Bug Fixes

- Add current path in service setup
  ([`00f2f3f`](https://github.com/cleanunicorn/drove/commit/00f2f3fd901accea53c75f5c5fba7e78391f499d))

- Chat wraps lines to display long lines
  ([`89e8f93`](https://github.com/cleanunicorn/drove/commit/89e8f938f2ebfabfcf1c3c381cf24c49013c297f))

- Correct exception handling in model selection
  ([`0e2e183`](https://github.com/cleanunicorn/drove/commit/0e2e18396bcd601d5b4dd1cef315449ac3b40b79))

- Disable observe web load-more while request is in flight
  ([`3887697`](https://github.com/cleanunicorn/drove/commit/388769757c277c4b170dd59d179d2045d95ba9e1))

- Exact local path takes priority over HF alias; resolve() is GGUF-only
  ([`9822244`](https://github.com/cleanunicorn/drove/commit/982224472cbcf88d81946d9efa1b8a013a5d3732))

Two correctness fixes raised in code review:

1. (P1) ModelStore.resolve() and find_root() now try the exact local path first and only fall back
  to HuggingFace alias resolution when no local directory/file matches. Previously _canonical() ran
  alias lookup unconditionally for any name containing '/', so an explicit org/repo directory could
  be silently redirected to a different model whose sidecar TOML claimed the same repo_id.

2. (P2) _find_primary() is now GGUF-only. The non-GGUF fallback (.safetensors/.bin/.pt) has been
  removed from the resolution path so ServerManager never receives a path that llama-server cannot
  open. list()/_add_entry() retains its non-GGUF fallback for display purposes.

Tests added for both invariants.

https://claude.ai/code/session_0176t6q7eDauxwVdvp1Bzw4P

- Fix python 2 syntax error in model selection
  ([`ba7d6f8`](https://github.com/cleanunicorn/drove/commit/ba7d6f8cbaf9292ac82baaed5394e6856a8cf0cd))

- Hold strong reference to fire-and-forget asyncio task to prevent GC
  ([`d359d79`](https://github.com/cleanunicorn/drove/commit/d359d7992c7cbaa107a1ecf421dfa6159eec5b1c))

The asyncio.create_task call for _save_observe_record was not storing the returned task, risking
  garbage collection before completion. Use a module-level set with add_done_callback(discard) to
  prevent this.

https://claude.ai/code/session_01Kz36GBmqJBK88PZoP4vqGx

- Move MD button to left
  ([`f453526`](https://github.com/cleanunicorn/drove/commit/f453526b02a8ed27b331ace9c2ebf749879ad0b1))

- Use exact quant_tag match in filter_by_quant to prevent F16/BF16 overlap
  ([`54e7097`](https://github.com/cleanunicorn/drove/commit/54e7097fb1cde8560e328040eb96098638b65251))

Substring matching caused selecting F16 to also include BF16 files. Switched to quant_tag()-based
  exact match (case-insensitive) to fix ambiguity.

Co-authored-by: Daniel Luca <cleanunicorn@users.noreply.github.com>

- **download**: Getting image model only selects one variant
  ([`b59a962`](https://github.com/cleanunicorn/drove/commit/b59a962220bf0e68a3a14fc3b98893a0f7ea2255))

- **observe**: Harden observe web load-more pagination
  ([`eb4cf33`](https://github.com/cleanunicorn/drove/commit/eb4cf3301be64dd233a863c230304d2f4711761d))

- Dedup records by id when appending a page so a record landing between loads cannot be inserted
  twice. - Capture the active search query in state at load time and reuse it for load-more, instead
  of re-reading the live input (avoids fetching a new query at a stale offset before the search
  debounce fires). - Guard `load-more-btn` lookup in renderList to match renderLoadMoreState.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

- **server**: Reload model reload if config changed
  ([`cd38bbd`](https://github.com/cleanunicorn/drove/commit/cd38bbdfad52a21dc1f7a31e4fe3cec87485dc4f))

### Build System

- Add service restart command
  ([`f4e0f65`](https://github.com/cleanunicorn/drove/commit/f4e0f658b1ff188f80de7ae09a83446dfb0b22ef))

### Documentation

- Add CHANGELOG.md and require changelog entries in AGENTS.md
  ([`bfe8b1f`](https://github.com/cleanunicorn/drove/commit/bfe8b1fa4ec18bbe3a27c6a7f1537fa89339555b))

Introduces a Keep-a-Changelog formatted CHANGELOG.md and updates AGENTS.md to require every change
  to land an entry under [Unreleased], in line with the planned Conventional Commits + semver
  release flow.

- Add info about service setup
  ([`7faa2c5`](https://github.com/cleanunicorn/drove/commit/7faa2c5b8376d4372be2263c772107069d2fff35))

- Add readme
  ([`6bab0a1`](https://github.com/cleanunicorn/drove/commit/6bab0a19abdd26a35799c8e16e9d3ab9e8056ecc))

- Clarify model management and concurrent execution in README
  ([`806f376`](https://github.com/cleanunicorn/drove/commit/806f376c3aee917d3c41d0e1abe3646da49833b9))

- Correct release branch references to master
  ([`b82b090`](https://github.com/cleanunicorn/drove/commit/b82b090d257482dc81040d7425a6e30cc8fbfd67))

The release docs described pushes to `main`, but the repo's default branch is `master`. Updated
  docs/deploy.md and the CHANGELOG pipeline entry so the prose matches the workflow trigger fixed in
  this PR.

Left intact: `[tool.semantic_release.branches.main]` (a TOML table name, not a branch) and its
  `(main|master)` match pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

- Document the release process in docs/deploy.md
  ([`8e973f7`](https://github.com/cleanunicorn/drove/commit/8e973f7e3568599fa6f3036dc073cca1167ae570))

Adds a maintainer-oriented walkthrough of the python-semantic-release pipeline: PR-title bump rules,
  required CHANGELOG handling, the on-merge workflow, a worked example, a maintainer checklist, and
  a troubleshooting section. Links the new page from the docs index.

- Fix per-model config key and model ID in examples
  ([`b2284ea`](https://github.com/cleanunicorn/drove/commit/b2284eac4557eef9586bd89f453b158721e89dde))

- Replace `context_size` with `ctx_size` to match the actual ModelConfig field (unknown keys are
  silently ignored via ConfigDict(extra="ignore")) - Replace misleading "any llama-server flag" note
  with the explicit list of supported ModelConfig keys - Fix SDK quickstart model ID:
  `unsloth/Qwen3-8B-GGUF` (the full repo ID that infer_local_name uses as the local name) instead of
  bare `Qwen3-8B-GGUF`

Co-authored-by: Daniel Luca <cleanunicorn@users.noreply.github.com>

- Improve readme
  ([`34a569f`](https://github.com/cleanunicorn/drove/commit/34a569fbef0282a9248e22c2f69c8c7d9e3c027b))

- Split model management lines
  ([`5383866`](https://github.com/cleanunicorn/drove/commit/5383866bb7c4415c9532f7f2a5be815dae7579db))

- Update PR title instructions to include link to Conventional Commits
  ([`744e73b`](https://github.com/cleanunicorn/drove/commit/744e73b31c4e5cbcfe9207c84bd6374274086a45))

- Update prerequisites section in README.md
  ([`35feda6`](https://github.com/cleanunicorn/drove/commit/35feda64087e0c5cfec9f60f0fb8cfcf2708c9f8))

- Update readme
  ([`a2665fd`](https://github.com/cleanunicorn/drove/commit/a2665fda1c4bcdb9b0ef13a194bb9e5a722fbe40))

- Update readme with syntax highlight
  ([`8caaa98`](https://github.com/cleanunicorn/drove/commit/8caaa98ab3e4b2669b18db7700ba0f8ec2824e55))

- fix header syntax highlight on TUI

- Update server status
  ([`7e526f3`](https://github.com/cleanunicorn/drove/commit/7e526f3692ce3b1bd1ed372f000b710b9815b830))

- **observe**: Add screenshot
  ([`5122fb3`](https://github.com/cleanunicorn/drove/commit/5122fb3f88e08f390b7301a897a828649a2290d4))

### Features

- Add --watch option to status command for continuous refresh
  ([`3fe6166`](https://github.com/cleanunicorn/drove/commit/3fe61660b9828e6d577e83e6215c36dba0d6e2b5))

- Add collapsible tool call sections in chat UI and enhance tool result rendering
  ([`66d789c`](https://github.com/cleanunicorn/drove/commit/66d789c24910728ce9da124d6e793711c22e66c2))

- Add CORS middleware to FastAPI applications in observe_web and proxy modules
  ([`1c79cda`](https://github.com/cleanunicorn/drove/commit/1c79cda1d61c31d915f3ab3352ab902565ab3f8b))

- Add feature for running multiple concurrent models on separate instances
  ([`57da516`](https://github.com/cleanunicorn/drove/commit/57da516a3c2575d7d5d84c4e06c97f20e318aa07))

- Add install script smoke workflow
  ([`5e81e67`](https://github.com/cleanunicorn/drove/commit/5e81e677d796824481fb0f7df07e019bf2303c5b))

- Add issue templates for bug and feature requests
  ([`56ababf`](https://github.com/cleanunicorn/drove/commit/56ababf4b9fd1894d0f19393671cce1520fa9f69))

- Add markdown toggle to observe web UI
  ([`e9ccf52`](https://github.com/cleanunicorn/drove/commit/e9ccf52c8e6550a0dc189f3f30e962736b25d103))

Co-authored-by: cleanunicorn <547012+cleanunicorn@users.noreply.github.com>

- Add max_loaded_models configuration and implement eviction of least-recently-used models
  ([`4a0d701`](https://github.com/cleanunicorn/drove/commit/4a0d70137cb76f134018618e4f347628c744845b))

- Add psutil dependency and implement conversion from HuggingFace format to GGUF
  ([`c0ee77c`](https://github.com/cleanunicorn/drove/commit/c0ee77cd5e7b71bdd54fd99239c838bb085a6b8b))

- Add search to observe
  ([`cea7cda`](https://github.com/cleanunicorn/drove/commit/cea7cdab431282fc778feb700db3cb24e69e7fa6))

- Add session management and terminal UI for vllama
  ([`9ec3051`](https://github.com/cleanunicorn/drove/commit/9ec3051022cfa747223de335fad1d414caccb9e3))

- Implemented session persistence with the ability to save, load, and list chat sessions. -
  Introduced a new Terminal UI (TUI) for interacting with the vllama server, allowing users to chat
  and manage sessions easily. - Enhanced model name autocompletion in CLI commands for better user
  experience. - Added configuration hot-reloading capabilities to the proxy server. - Updated
  configuration schema to include a sessions directory and TUI theme settings. - Refactored server
  manager to utilize the new idle timeout configuration.

- Chat accepts openai compatible endpoint and model listing
  ([`170ed09`](https://github.com/cleanunicorn/drove/commit/170ed09ff37680d51523e4066430ae6893a26a24))

- Chat prompts to select model if none specified
  ([`c504783`](https://github.com/cleanunicorn/drove/commit/c50478335a488b6235df2ba5d1531fbf1d391157))

- format with ruff

- Create initial version of llama.cpp handler
  ([`4afea20`](https://github.com/cleanunicorn/drove/commit/4afea20679858dbd79167192ab539d0ea12825c5))

- Display server stats, find available open port for llama-server
  ([`a416892`](https://github.com/cleanunicorn/drove/commit/a416892876c34d5a46fa2fd569a4fb82fa937367))

- Enhance auto-scroll functionality in chat app to allow forced scrolling
  ([`544cf1b`](https://github.com/cleanunicorn/drove/commit/544cf1be5aa1467f712017b190ee7f54f3682b32))

- Enhance model config command to display effective configurations with sources
  ([`a966124`](https://github.com/cleanunicorn/drove/commit/a966124726243a098f423cf6aa08db0a7df9a1b5))

- Enhance process statistics reporting and token usage metrics
  ([`d6b4ce8`](https://github.com/cleanunicorn/drove/commit/d6b4ce8b1ef9b32fe168df7281ffb1a637305a3e))

- Enhance server management and proxy handling for multiple models
  ([`fdd484e`](https://github.com/cleanunicorn/drove/commit/fdd484e3db1685daae96763a8ca25ec848d19043))

- Enhance server status output to support multiple models and improved process metrics
  ([`2dd4320`](https://github.com/cleanunicorn/drove/commit/2dd43204f60f703b83a9d9ac98feced565c9c8f9))

- Enhance status command to support continuous refresh without exiting on connection errors
  ([`105519c`](https://github.com/cleanunicorn/drove/commit/105519cf418dddf7d27fa26f705b8be2e42cf3e4))

- Implement global config
  ([`f7a22ab`](https://github.com/cleanunicorn/drove/commit/f7a22ab05041d00ab4ae01b7a5fc1a3588a9b982))

- Implement request tracking in ServerManager to prevent premature shutdown
  ([`0c02bdb`](https://github.com/cleanunicorn/drove/commit/0c02bdbd7a263b55475574bafd343bd8c3facd30))

- Implement server management commands for starting, stopping, and restarting the vllama server
  ([`d4d3c93`](https://github.com/cleanunicorn/drove/commit/d4d3c933bde9936c3c9aed0fdc4bab8d0e503831))

- Move status command from `vllama status` to `vllama server status`
  ([`2427f2b`](https://github.com/cleanunicorn/drove/commit/2427f2bb92c022775f7ff498f8a3dbb84a13281b))

- Prompt quant selection when more gguf files are found in the repo
  ([`7b309be`](https://github.com/cleanunicorn/drove/commit/7b309bec9418e2e146d737bde8b2c091e90d9d92))

- Refactor type annotations and add tool execution functionality
  ([`c65e347`](https://github.com/cleanunicorn/drove/commit/c65e3470950836714338a47cd3fe81592218507b))

- Remove stop and restart commands from server management
  ([`2220d5c`](https://github.com/cleanunicorn/drove/commit/2220d5c9a0b66baf31b3291aefe73c1c9b45f343))

- Rename CLAUDE.md to AGENTS.md
  ([`a622079`](https://github.com/cleanunicorn/drove/commit/a622079ece589b21f0553d734643cc4d67e84780))

- Restart model server when config changes
  ([`e667f5b`](https://github.com/cleanunicorn/drove/commit/e667f5b951a2193ce756dd30793ad14cc679f405))

Track config file mtimes in _ModelInstance at startup. On each ensure_running() call, compare
  current mtimes against the stored snapshot. If the per-model or global TOML has been modified: -
  restart immediately when the model is idle - set needs_restart so the idle watcher stops it as
  soon as all in-flight requests drain

Closes #17

Co-authored-by: Daniel Luca <cleanunicorn@users.noreply.github.com>

- Update flash attention configuration to accept string values
  ([`cfc8d2a`](https://github.com/cleanunicorn/drove/commit/cfc8d2a528c053a35e6e7102b540c0311b9b42b5))

- **config**: Add startup timeout configuration for llama-server health check
  ([`76c614a`](https://github.com/cleanunicorn/drove/commit/76c614ac19204df3a0ebf1aa1154b6d155c69962))

- **models**: Add capability detection for vision models
  ([`5083dab`](https://github.com/cleanunicorn/drove/commit/5083dabf886b4c9ce427b7a179587040bb4ca143))

- **models**: Enhance model directory scanning to support namespaced models
  ([`8c35024`](https://github.com/cleanunicorn/drove/commit/8c3502452aa08251080bef6ced4edbe84809db25))

feat(downloader): improve local model name inference with quant support

fix(model_config): update alias resolution to scan TOML files recursively

- **models**: Enhance model resolution and add support for HuggingFace references
  ([`4916279`](https://github.com/cleanunicorn/drove/commit/49162794d832964c8a3e88cd3ae9bf33d49168c5))

- **observe**: Add paginated request list with load more
  ([`9192958`](https://github.com/cleanunicorn/drove/commit/9192958dd2a08a9da410e188254a19a22436c380))

- **observe**: Add syntax highlighting for request and response data
  ([`41b7f92`](https://github.com/cleanunicorn/drove/commit/41b7f92295dddd9f5814413444708d8065b797da))

- **observe**: Add TUI for browsing logged API requests and responses
  ([`64e03c6`](https://github.com/cleanunicorn/drove/commit/64e03c67dd10f9c900f851ff5c274976de840ffd))

- **observe**: Enhance list_records to support namespaced model directories
  ([`d7dab9a`](https://github.com/cleanunicorn/drove/commit/d7dab9a5ea575133c075309666a3cf2a571d693d))

- **observe**: Implement JSON tree visualization for request and response bodies
  ([`2f8a1c5`](https://github.com/cleanunicorn/drove/commit/2f8a1c5781a722249ee2823a006c024700defdcb))

- **observe**: Implement request/response logging and TUI for browsing logs
  ([`6929e7b`](https://github.com/cleanunicorn/drove/commit/6929e7bc661e66153fe56b162461e79fcb6ea050))

- Added observation logging functionality to capture API requests and responses. - Introduced a new
  command `observe` in the CLI to browse logged API requests and responses. - Created
  `ObserveRecord` and `ObserveContext` classes to structure logged data. - Implemented functions to
  save, load, and list observation records. - Developed a Textual-based TUI for displaying observed
  records with detailed views. - Integrated observation logging into the proxy to record request and
  response details. - Added tests for the observe module and its integration with the proxy.

- **observe**: Implement web UI for browsing logged API requests and responses
  ([`14fc561`](https://github.com/cleanunicorn/drove/commit/14fc561133b1db58e51f24ca6f6e886ba881fb62))

- **server**: Enhance ensure_running to support atomic request slot claiming
  ([`97aa71b`](https://github.com/cleanunicorn/drove/commit/97aa71ba1fc29fda0d4d88540e2f6cfab060db1e))

- **server**: Enhance stderr handling and auto-configure mmproj paths
  ([`1991ee5`](https://github.com/cleanunicorn/drove/commit/1991ee57ec8479cd749fc3dcf67771485eff9c43))

- **server|downloader**: Enhance server and file download logic for multimodal projections
  ([`640bbb6`](https://github.com/cleanunicorn/drove/commit/640bbb6937c1524d9f41be6bd9059d4fd09ae495))

- **tests**: Add test for list_records to find namespaced models
  ([`9cec944`](https://github.com/cleanunicorn/drove/commit/9cec94417042c7daf6df7f07f7b6741768965704))

### Performance Improvements

- **observe**: Page records from disk instead of parsing all
  ([`2bf0f79`](https://github.com/cleanunicorn/drove/commit/2bf0f791fa00d4e10bfa74357f2c274e543b3280))

`list_records` loaded and JSON-parsed every record on disk per /api/records call, so large logs paid
  a full scan just to show one page.

- Add `list_records_page(observe_dir, model, offset, limit)` which sorts record files by name (the
  id timestamp prefix encodes chronological order) and reads only the requested window, returning
  the page plus the total count. - Extract dir resolution into `_record_dirs` and path collection
  into `_record_paths`, shared by `list_records` and `list_records_page`. - `/api/records` uses
  `list_records_page` when no search is active; search still loads the full set since filtering
  needs every record.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

### Refactoring

- Extract ModelStore to unify model resolution
  ([`84faf43`](https://github.com/cleanunicorn/drove/commit/84faf438bfa967a47fe00d4a1988ab63c57449f4))

Both server_manager and cli/models had independent model-path lookup implementations with subtly
  different behaviour: the CLI used rglob() (recursive) while the server used iterdir()
  (non-recursive). This meant a model whose GGUF sits one directory level below the model root would
  be visible to the CLI but fail to launch in the server.

ModelStore is now the single authority for all filesystem model operations — resolve(), find_root(),
  list(), complete() — and both callers delegate to it. The rglob-based resolution is used
  everywhere, closing the divergence bug.

153 lines of duplicated helper functions removed across two files.

https://claude.ai/code/session_0176t6q7eDauxwVdvp1Bzw4P

- Observe web
  ([`3c09bc4`](https://github.com/cleanunicorn/drove/commit/3c09bc47434083913d890941f747a722df1a0b97))
