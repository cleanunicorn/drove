# CHANGELOG

## [Unreleased]

### Fixed

- Scoped chat resume and session selection to the active endpoint, with explicit
  assignment for older conversations that had no endpoint metadata.
- Restored saved system prompts when resuming conversations and cleared temporary
  tool approvals when loading another session; rejected `--resume --system`
  instead of silently ignoring the supplied prompt.


## v0.14.0 (2026-10-28)

### Bug Fixes

- **chat**: Align the approval scope line and mark the reason
  ([`2d7df0f`](https://github.com/cleanunicorn/drove/commit/2d7df0f892dcfce6d30622b523ad06f832482610))

The #tool-scope label lacked the "message" class its siblings carry, so it started two cells left of
  the reason line and ran straight into "Arguments:". Give it the class, and give #tool-reason a
  $warning rule (the project's attention colour) so the reason stands out from the boilerplate
  intro. Existing tokens only.

Belongs to: #154.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Cap old tool results when a session is loaded from /sessions
  ([`78f1c32`](https://github.com/cleanunicorn/drove/commit/78f1c322db083b5e4c3ea9d2f57a9a4da3876e4e))

Only constructor resume capped stored tool results; choosing an older conversation in /sessions put
  an oversized result back in history and in the next model request. Cap it at that load too.

Refs review A-correctness-1.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: End a pending mid-fetch approval on stop or quit
  ([`ef258b8`](https://github.com/cleanunicorn/drove/commit/ef258b8a9e2a15eee5db360f118112308706bc94))

The worker's wait for a mid-fetch origin approval only noticed a closed loop, so a stop or quit left
  the dialog open and the turn generating until someone answered. The wait now also watches
  stop_requested and cancels the approval, which dismisses its dialog and refuses the fetch.

Refs review A-correctness-2.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Judge NAT64, 6to4 and other IPv4-in-IPv6 forms by the IPv4
  ([`d3a90dd`](https://github.com/cleanunicorn/drove/commit/d3a90ddc257e960234f9bc457e4abbe94515d026))

64:ff9b::/96 is globally routable to ipaddress, so a NAT64 address of 169.254.169.254 or 10.0.0.1
  passed the internal-address check. Decode IPv4-mapped, IPv4-compatible, NAT64 (well-known and
  local-use /96 layout), 6to4 and Teredo addresses and apply the IPv4 rule.

Refs review B-security-2.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Never leave a tool worker waiting on a closed loop
  ([`7e54ab2`](https://github.com/cleanunicorn/drove/commit/7e54ab28421d77735b38ef9ddbf05a2ccb912807))

Poll the mid-fetch approval so a worker returns when the event loop is gone, and drop the per-call
  grant when a call is denied.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Refuse out-of-root reads before revealing whether they exist
  ([`2a82b01`](https://github.com/cleanunicorn/drove/commit/2a82b0177f2be0d3ed28012ee4e8e827b771af2b))

read_file, list_files and search_files ran their exists/is_file/is_dir checks before _check_roots,
  so a call outside the approved roots got "not found" or "not a file" instead of the roots refusal,
  which leaks whether the path exists. Check the roots first in all three.

Belongs to: #154.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Scope fetch approvals to scheme, host and port
  ([`07b2fd3`](https://github.com/cleanunicorn/drove/commit/07b2fd37f9dead66e32e53dd5c9566e6c85b83bc))

A fetch grant covered every scheme and port of a host, so approving https://example.com also allowed
  http://example.com and https://example.com:8443 (another service) without asking, directly or
  through a redirect. Grants, session approvals, the execution context, redirect checks and
  internal-address approvals now use the canonical origin scheme://host:port (default port spelled
  out). Host-only and non-canonical saved grants are ignored, so nothing widens silently.

Docs also note the IPv4-in-IPv6 rule from the previous commit.

Refs review B-security-1.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Treat local-use NAT64 addresses as internal
  ([`7c36657`](https://github.com/cleanunicorn/drove/commit/7c36657dc81957f5a427765810c5368d4f854f81))

64:ff9b:1::/48 was decoded as if the IPv4 address sat in the last 32 bits, but local-use prefixes
  may be /48 to /96 with the IPv4 bits elsewhere, so a private target with a public-looking suffix
  skipped internal approval. Keep decoding the well-known 64:ff9b::/96 and block the whole local-use
  range unless the user approves the origin.

Refs review B-security-3.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Warn when an allow-always approval cannot be saved
  ([`5b64ddb`](https://github.com/cleanunicorn/drove/commit/5b64ddb20d80efbfabd515f3ded72bb278670a0c))

The persistence failure was swallowed silently. Since #148's set_config_value validates the whole
  file, any invalid key elsewhere now blocks the save, so the user chose "Allow always", saw
  nothing, and was asked again next launch. Show a warning notification; the session grant still
  applies.

Belongs to: #154 (made more likely by #148).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **cli**: Keep logging off the terminal while a Textual app runs
  ([`aa1d19e`](https://github.com/cleanunicorn/drove/commit/aa1d19e28388080e23825ac0a2df77aacfffc002))

The root callback's basicConfig binds a StreamHandler to the real stderr before any command runs.
  Textual redirects sys.stderr while it owns the screen, but that handler keeps the original stream,
  so httpx's INFO "HTTP Request" lines were drawn over drove chat next to the input box.

Swap the root handlers for a single TextualHandler (no stdout/stderr fallback) around every Textual
  run — chat, the initial model picker, observe, and the quant picker — and restore them afterwards.
  Records, including DEBUG under `drove -v`, go to Textual's log instead; non-TUI commands and drove
  server log to stderr unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **cli**: Require :QUANT for unattended multi-quant downloads
  ([`2f0337e`](https://github.com/cleanunicorn/drove/commit/2f0337e107bbfc2c225b7db954662e3740e88217))

`drove models download <repo> --yes` on a repository with several GGUF quantizations now fails
  before touching the disk and asks for org/repo:QUANT instead of downloading every variant.
  Rerunning a download whose files all landed but whose record is still incomplete finishes the
  record instead of exiting early, `models info` reports an incomplete download with its reason, and
  the retry hint after an all-variants GGUF download keeps the picker. Consumer tests cover
  /v1/models, chat routing and agent registration.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **config**: Name the variable for invalid members of JSON env values
  ([`682946c`](https://github.com/cleanunicorn/drove/commit/682946c41fc1bd659a7e821c4981b3bd38e188a4))

An invalid element in an otherwise valid DROVE_ALLOWED_TOOLS list, or an invalid field in a
  DROVE_LLAMA_SERVER object, was reported as "DROVE_*" because any dotted error location was treated
  as a llama_server leaf. Attribution now starts from the root field and also considers the
  whole-field variable.

Findings: A:correctness-1, B:correctness-2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **config**: Report invalid paths as config errors, not tracebacks
  ([`0a34382`](https://github.com/cleanunicorn/drove/commit/0a343823688634b4cd0e5899bf052f3f5ce66484))

`models_dir = 123` (TypeError) or a `~user` path that cannot be expanded (RuntimeError) escaped the
  readable-error boundary because only ValueError becomes a pydantic validation error. The path
  validator now raises ValueError, so the file or DROVE_* variable and key are named and `drove
  config <key> <value>` can repair it. The same unexpandable `~` in --config or DROVE_CONFIG now
  gives a readable error too.

Finding: B:correctness-1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **config**: Save only the changed key and explain broken config
  ([`0659bda`](https://github.com/cleanunicorn/drove/commit/0659bdabffc8dd1e907524db70938810f6362977))

Every config write started from the env-loaded Config and dumped all fields, so a one-off DROVE_*
  override (and every default) became permanent. Writers now patch one key in the raw TOML mapping
  and validate the merged file with an env-free settings class; init writes true defaults.
  Config.save is removed so no caller can reintroduce the bug.

Loading wraps TOML, validation and env-parse failures in ConfigError, naming the file or DROVE_*
  variable and the key; the CLI prints it and exits 1. `config` and `init` no longer load the config
  up front, so `drove config <key> <value>` can repair a broken key.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **downloader**: Refuse non-GGUF repos and record download completion
  ([`99770e9`](https://github.com/cleanunicorn/drove/commit/99770e909fb58560bac7b5064a37bfca8e6ba0f0))

Text and vision downloads take GGUF files only; a repository with no GGUF or ONNX weights is refused
  with a message that drove does not convert weights. Every download now writes its record
  incomplete before the first transfer and complete after the last, so an interrupted download is
  never listed or served, and two repository files that would flatten onto one local path are
  refused before anything is written.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **model-store**: Apply a symlinked primary's target records (correctness-1)
  ([`e5f2f40`](https://github.com/cleanunicorn/drove/commit/e5f2f4069d6c29b98a21e2378dc707a356b3ab01))

A contained file symlink (alias/model.onnx or alias.gguf pointing into another model's directory)
  reached an unfinished or stale download without consulting its record, because records were looked
  up only along the lexical path. Also walk the resolved target's ancestry, and apply the
  split-shard rule to the target, at every call site. No-record links, a symlinked models directory
  and old downloads keep working.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **model-store**: Honour every download record above a primary (correctness-1)
  ([`7b63781`](https://github.com/cleanunicorn/drove/commit/7b637813cc278119d9b69a865504832fe159fdc1))

Completeness was checked only at the directory the requested name pointed at, so resolving a parent
  namespace ("org") reached an unfinished child download ("org/asr") without consulting its record.
  Check every record between the models directory and the selected primary, for list, resolve and
  flat files alike.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **model-store**: List and serve only complete, servable models
  ([`d9b504e`](https://github.com/cleanunicorn/drove/commit/d9b504edc1e16d59ac4f1dfa9be8f70a2a46a918))

Remove the arbitrary-file primary fallback so Hugging Face cache metadata and raw
  safetensors/.bin/.pt weights are never listed, completed, advertised or integrated. Hidden paths
  never become a primary, a split GGUF needs every shard, and a per-model download record
  (.drove-download.json) must say complete with every file at its expected size. Downloads made
  before the record existed are judged by their files. Incomplete models raise IncompleteModelError
  (a FileNotFoundError).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **model-store**: Require a download record to cover its primary (correctness-2)
  ([`dc22b96`](https://github.com/cleanunicorn/drove/commit/dc22b961ee301abdd21e9d7d2cba87583b13d6bc))

A complete record only checked the files it listed, so a stale or empty record let an unrecorded
  (possibly partial) file become the served primary. The selected primary must now be listed by
  every record above it; a TTS model's generated marker is the one exception, since the record lists
  its weights instead. Models without a record are unaffected.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **models**: Ignore nested TTS markers regardless of suffix case
  ([`5fb6af0`](https://github.com/cleanunicorn/drove/commit/5fb6af086c0aa3a03ae1ddd5d8b31913abde8487))

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **models**: Ignore undecodable download sidecars when picking weights
  ([`8479829`](https://github.com/cleanunicorn/drove/commit/8479829cfd077e40192d21420928c3be9a0dfd67))

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **models**: Keep resolve ignoring nested TTS markers
  ([`8683d21`](https://github.com/cleanunicorn/drove/commit/8683d2111fc188da770b083a9cc4957efdf4d93b))

Unifying primary selection had made resolve() pick a nested .tts over a GGUF; restore the previous
  top-level-only behavior and pin it with tests.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **models**: Limit projector exclusion to GGUF files
  ([`1a6d515`](https://github.com/cleanunicorn/drove/commit/1a6d51599cbaa84472e38deb67637ce7f4d00990))

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **models**: Name downloads interrupted before any weights arrived
  ([`1a42ae6`](https://github.com/cleanunicorn/drove/commit/1a42ae6e4782113621e9231ec2e5b619d2830ba3))

The re-check found ModelStore.incomplete() missed the commonest interruption: the downloader writes
  the incomplete record before the first transfer, so a failure there leaves a root holding only the
  record and the transfer cache. With no primary, it was neither listed nor reported, and drove
  models list still said "No models found". Treat a directory with a download record as a model
  root, and report it as incomplete when the record says so.

Belongs to: #152.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **models**: Select vision weights instead of mmproj projectors
  ([`c4d83f4`](https://github.com/cleanunicorn/drove/commit/c4d83f4d8a6b4238e46b1dd725ff172c3776cc6e))

Primary selection had two case-sensitive copies that never excluded mmproj files, so a projector
  sorting first became --model and its sidecar was missed. One selector now serves list and resolve,
  skips projectors (shared MMPROJ_RE with the downloader), prefers the download sidecar's first
  weights entry, and treats projector-only directories as not a model.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>

- **observe**: Hold the 1 MiB record cap for oversized metadata
  ([`eb3e8a2`](https://github.com/cleanunicorn/drove/commit/eb3e8a2142d58b20677f23831720a52bb3f1ca0e))

The encoder shrank only bodies and returned an oversized document once its budget ran out, so a long
  request path could exceed the cap. The endpoint and method are now bounded with a truncation
  marker, a record that still does not fit drops its headers and bodies with a marker, and one that
  cannot fit at all is refused rather than written.

Refs B-correctness-1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **observe**: Never prune or write through a swapped directory
  ([`7de42b3`](https://github.com/cleanunicorn/drove/commit/7de42b3760218dcac1dcfe6edf24ff38ed9f093b))

Retention checked containment by path and then scanned and unlinked by path, so a model directory
  replaced with an outward symlink mid-pass could redirect deletions outside observe_dir. Retention
  now walks by directory descriptor with O_NOFOLLOW and stats and unlinks each file relative to its
  pinned directory. save_record opens model directories the same way, and rejects record ids that
  are not plain file names.

Refs B-security-1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **observe**: Prune namespaced records that share a parent with flat ones
  ([`5543005`](https://github.com/cleanunicorn/drove/commit/5543005b873b92abbe547639b0be714d3116bfc6))

Retention reused the listing helper, which searches a directory one level deeper only when it holds
  no records, so org/repo records were never pruned while org had records of its own. Retention now
  visits every model directory and each of its child directories.

Refs A-correctness-1, B-correctness-2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **observe**: Skip records the retention pass deletes mid-read
  ([`a55cc0e`](https://github.com/cleanunicorn/drove/commit/a55cc0ed31da89372bc49822e4d810552dc322b4))

The hourly retention pass can unlink a record after a reader enumerated it but before it opened it.
  list/page/search readers and the web detail endpoint did not catch FileNotFoundError, so the web
  list returned 500 and the TUI showed "Unable to read requests". Treat a vanished record as a
  concurrent deletion: skip it (404 for the detail endpoint).

Belongs to: #153.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **proxy**: Bound observe capture and write records off the event loop
  ([`724b209`](https://github.com/cleanunicorn/drove/commit/724b2092e5614fbaf67633e45580bf8e25a5e2f2))

The proxy no longer joins every response in memory: it keeps the last 1 MiB for usage parsing (none
  for audio) and, only when observing a text response, the first 256 KiB. Request bodies are
  snapshotted to the same bound. Records are built, encoded and written in worker threads by a
  per-app writer that drops records past 64 pending writes and drains on shutdown. Expired records
  are pruned at startup and hourly using the current observe_max_age_days.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **proxy**: Close the upstream response when cancelled mid-handoff
  ([`d3442bf`](https://github.com/cleanunicorn/drove/commit/d3442bfccac0cd55b6927b990d6fb8c40169d185))

If the handler was cancelled while the send race wound its watcher down after the send had already
  won, the helper exited without returning or closing the upstream response, leaving its connection
  open. Treat the response as handed over only once the helper returns, and close it otherwise in a
  shielded cleanup so a repeated cancellation cannot interrupt the close.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **proxy**: Explain incomplete downloads instead of reporting not found
  ([`fc3fb8d`](https://github.com/cleanunicorn/drove/commit/fc3fb8de6b5c8cf838858030278e79de0ed1764e))

IncompleteModelError subclasses FileNotFoundError, so the fixed startup mapping from #146 answered
  "not found, run drove models list" for #152's incomplete downloads, which that list deliberately
  hides. Map it first to a fixed 404 that says the download is incomplete and how to finish it,
  still without paths or exception text.

Belongs to: #152 (interaction with #146).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **proxy**: Keep the models directory out of unknown-model 404s
  ([`c2e0567`](https://github.com/cleanunicorn/drove/commit/c2e0567cb8973d3caa8fd832a03b367e555c00f3))

The 404 detail was the store's exception text, which names models_dir. Return a fixed message built
  from the validated model name instead; the original stays in the server log.

- **proxy**: Keep token usage for non-streaming JSON over 1 MiB
  ([`c9ee25b`](https://github.com/cleanunicorn/drove/commit/c9ee25b4377395fb803c55debf587f65eadac4e4))

The bounded usage tail cut large JSON responses at the front, so the whole-document parse failed and
  their usage was lost. When neither the JSON nor the SSE path finds usage, the trailing usage and
  timings objects are now parsed from the retained tail.

Refs A-correctness-2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **proxy**: Map startup failures deliberately and keep backend stderr private
  ([`46a4d7f`](https://github.com/cleanunicorn/drove/commit/46a4d7f66e4df9521ad43720d7519ccb39fab627))

Every exception escaping ensure_running now settles the request counters and maps to a fixed, short
  response: 503 for a missing or unlaunchable backend (new BackendUnavailableError) and failed
  health checks, 500 for a broken model config or an unexpected error, 404 for an unknown model.
  Config and backend validity are checked before an idle restart or an eviction, so a broken sidecar
  no longer costs a healthy model. Backend stderr moves from the exception message to its notes and
  the server log.

- **proxy**: Stop forwarding requests after the client disconnects
  ([`44b22a1`](https://github.com/cleanunicorn/drove/commit/44b22a15cea424d0867e908e117905b5c77b44ee))

A client that disconnected while its model was cold-starting still had its request sent upstream,
  keeping the model busy until the backend answered. A cancelled handler also skipped the
  Exception-only cleanup, leaving the active-request counts raised.

Check for a disconnect once the slot is claimed, race the upstream send against the client's
  http.disconnect (cancelling the send if the client leaves first), and settle the claim exactly
  once on any BaseException. Mid-stream disconnects are unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **server**: Log a failed backend's stderr once
  ([`f807309`](https://github.com/cleanunicorn/drove/commit/f807309a869ac65288a2bba9ed0065a337a233f7))

ServerManager._start and proxy._startup_error both formatted the exception notes (backend stderr)
  into their log lines, and the proxy also attached the traceback, which prints the notes again.
  Keep one owner: the proxy, the manager's only caller, logs once with exc_info (the traceback
  carries the notes); the manager logs a one-line failure. Removes the duplicated __notes__
  formatting.

Belongs to: #146.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **server**: Reload the model config after eviction drains
  ([`2c63da6`](https://github.com/cleanunicorn/drove/commit/2c63da641d6e6f44fd44711d453db9434496fe1d))

The config snapshot taken before eviction could go stale while a busy model drained, launching a
  backend with old settings but recording the new sidecar mtime, so the edit was never applied.
  _start reads the config again; the pre-eviction check still keeps a broken sidecar from evicting
  anything.

- **sessions**: Save atomically, skip unreadable files, unique ids
  ([`0a54bec`](https://github.com/cleanunicorn/drove/commit/0a54bec17e7dca66ff93adda4fe5841c1e0e5858))

Write each session through a same-directory temp file and os.replace so a failed save never
  truncates the previous copy. Validate loaded files and skip unusable ones with a warning so
  --resume and /sessions keep working. Give new sessions microsecond ids with a random suffix so
  chats started at the same instant get separate files, and validate session ids so a stored id
  cannot point a save outside the sessions directory.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **sessions**: Skip session files whose messages chat replay cannot show
  ([`39dea95`](https://github.com/cleanunicorn/drove/commit/39dea95f2bcff978beb23d91dce44cf3b2c22f3b))

A tool message without a string tool_call_id (or a non-text content, thinking, or malformed
  tool_calls entry) passed load validation, so --resume picked the file and history replay crashed
  with KeyError. Validate every message field replay reads so such files are skipped.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **sessions**: Skip session files with null tool-call name or arguments
  ([`178f52f`](https://github.com/cleanunicorn/drove/commit/178f52f8894fb45e60d78e21ccbc6de6d68f217a))

Replay previews tool-call arguments as text, so an explicit null crashed it after --resume picked
  the file. Earlier drove versions always stored both as strings, so rejecting non-strings keeps old
  sessions loadable.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **sessions**: Skip session files with roles chat replay cannot show
  ([`7d8fec2`](https://github.com/cleanunicorn/drove/commit/7d8fec209d36d84dc4b6cc0ab1075e1243a80cd1))

Replay turns a message role into a CSS class, so an empty or unknown role string crashed it after
  --resume picked the file. Accept only system, user, assistant and tool, the roles drove has ever
  written, so sessions from earlier versions still load.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **stats**: Bound proxy speed and TTFT samples
  ([`7878313`](https://github.com/cleanunicorn/drove/commit/787831316daaf3f7b21d29bd0f11ccf3d7645628))

Keep at most 1024 recent samples and compute averages from lifetime running sums, so memory and
  /status cost no longer grow with traffic while the reported averages stay the same.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **tui**: Rely on unique session ids and log skipped session files
  ([`cc8c998`](https://github.com/cleanunicorn/drove/commit/cc8c998ee0c3663b154b207d912fa798d284d109))

The switch-model suffix workaround is no longer needed now that every new session id is unique.
  /sessions routes skipped-file warnings to the Textual log instead of the terminal the app owns.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Documentation

- Describe startup error statuses and private stderr
  ([`978bb5b`](https://github.com/cleanunicorn/drove/commit/978bb5b3416e910635c052b274b95cfb1953345e))

- Document latest_session and parse_grants
  ([`0c56de0`](https://github.com/cleanunicorn/drove/commit/0c56de0d64451773a432a7d1882b818d164dd025))

Both are new public helpers whose siblings are documented; say what each returns and that each
  silently skips unreadable or invalid input.

Belongs to: #150 (latest_session) and #154 (parse_grants).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Note chat session recovery and changelog for #119
  ([`e425567`](https://github.com/cleanunicorn/drove/commit/e425567960d92e67de0f977af925a3d148a3903f))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- Remove the unused converter and document GGUF-only downloads
  ([`7897ad1`](https://github.com/cleanunicorn/drove/commit/7897ad1cb08b34dcdd02a57bfc553d3d0aaad675))

drove never converts models: delete the unused HF-to-GGUF converter module and its project-map
  entry, and document that non-GGUF repositories are refused, that incomplete downloads stay hidden
  until finished (and how older downloads are treated), and that --yes asks for :QUANT on a
  multi-quant GGUF repository. Add the changelog entries for the fix.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **agents**: List cli/tui_logging.py in the project map
  ([`56bf019`](https://github.com/cleanunicorn/drove/commit/56bf01980f485ebdd2e07b5c918d28aa86720699))

#149 added the module every Textual entry point goes through; the map listed the batch's other new
  modules but not this one.

Belongs to: #149.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **changelog**: Cover the fix series' refactors and docs, in past tense
  ([`12fb7fe`](https://github.com/cleanunicorn/drove/commit/12fb7fe29034b0b1be476db617ceea6512a72818))

The re-check noted that the refactor, documentation and test commits had no Unreleased entry,
  although AGENTS.md asks for one for every change, and that the new bullets used the present tense.
  Add one Changed bullet for the shared-logic refactors and one for the docs and tests, and put this
  series' bullets in the past tense.

Belongs to: batch (spans #146, #148, #149, #150, #152, #153, #154).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **changelog**: Fold the second Unreleased section into the first
  ([`c8f39c0`](https://github.com/cleanunicorn/drove/commit/c8f39c0060fa7f378555fb382f9901d61b5f2025))

#150's clean auto-merge left its own "## [Unreleased]" section below the batch's. Move its three
  Fixed bullets and one Security bullet into the matching groups of the single Unreleased section;
  every bullet kept.

Belongs to: #150.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **changelog**: Note disconnect-aware forwarding
  ([`ed8a75c`](https://github.com/cleanunicorn/drove/commit/ed8a75c83e45bdd45b52dbfe26834e19257eae6a))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **changelog**: Use past tense and limit repair claim to bad values
  ([`f123488`](https://github.com/cleanunicorn/drove/commit/f1234887c8714346eaab73f9c0140561c0a5b1b2))

The entry said `drove config` could repair a file with broken TOML syntax, which it refuses by
  design; it also used present tense, against AGENTS.md golden rule 7. docs/cli.md now says the
  same.

Findings: A:docs-1, B:docs-1, A:conventions-1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Document tool approvals, grant scopes and limits
  ([`db38acb`](https://github.com/cleanunicorn/drove/commit/db38acb7c71187f1e5165ad7c13304218981dd6f))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **cli**: Explain where logs go while a TUI runs
  ([`888dbc4`](https://github.com/cleanunicorn/drove/commit/888dbc46981ae54b6c0b82e67da44f684f1bd792))

Note the root placement of -v, that records go to Textual's log during chat, observe and the
  pickers, and how to view them with textual console; add the Unreleased Fixed entry for #141.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **config**: Describe single-key saves and config error repair
  ([`d90b59e`](https://github.com/cleanunicorn/drove/commit/d90b59ef8b12daa975dd21a13cb8a68ec5e6cdab))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **config**: Describe the allowed_tools entry format
  ([`07057cd`](https://github.com/cleanunicorn/drove/commit/07057cd625b975b946867cecc542efa21a901e8b))

Its entries became versioned, scoped grant tokens in #154 while the neighbouring fields carry inline
  comments; say what an entry is.

Belongs to: #154.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **config**: Say add_allowed_tool stores grant tokens, not tool names
  ([`6d06b7e`](https://github.com/cleanunicorn/drove/commit/6d06b7e3d43b8b9cb6b6624d85ce5e7449c6878c))

After #154 its only caller passes Grant.token(), and the policy ignores bare names. Rename the
  parameter to entry and say so in the docstring.

Belongs to: #154 (helper added by #148).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **downloader**: Say why _check_targets calls _expected_files
  ([`e565631`](https://github.com/cleanunicorn/drove/commit/e56563148be817231c91f355f31d40835a103485))

The bare call's result is discarded on purpose: it raises for colliding flattened paths before any
  I/O. Say so, so a tidy-up does not delete the guard.

Belongs to: #152.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **observe**: Document retention, redaction and record limits
  ([`ab3f2f4`](https://github.com/cleanunicorn/drove/commit/ab3f2f43c940f64377b0f31bdfebfb27d7e7718b))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Features

- **chat**: Add a pure approval policy and output cap for chat tools
  ([`4a75f89`](https://github.com/cleanunicorn/drove/commit/4a75f893de68f0a745b2904579b4abc02224f704))

Approval decisions (reads outside the working directory, every read on a remote endpoint, per-call
  shell and write, host-scoped fetch) and the 64 KiB UTF-8 output cap live in tool_policy.py so they
  can be tested without the TUI.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Bound tool output, shell processes and fetch destinations
  ([`8f82673`](https://github.com/cleanunicorn/drove/commit/8f82673da44f04c7da0d7cc28f0525d853fc6a99))

- Every tool result is capped to 64 KiB; read_file reads 200 lines by default and never more than
  the cap, list/search stop at the cap. - search_files checks the name first and skips large, binary
  and undecodable files; reads honour the approved roots from a ToolContext. - Shell timeouts are
  clamped to 1..600 s, commands run in their own process group which is killed on timeout and after
  exit; background tasks are limited to 4 running and 16 kept for 10 minutes. - fetch_url follows up
  to 5 redirects, checks every DNS answer of every hop, refuses internal addresses unless approved,
  pins the connection to the vetted address (Host and SNI kept) and ignores env proxies.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Scope tool approvals and cap results in history and sessions
  ([`91c747b`](https://github.com/cleanunicorn/drove/commit/91c747bdc42607370448fca2ef29bbdc68cd643b))

- The chat decides approvals through tool_policy: reads outside the working directory ask, every
  read asks on a non-local endpoint, shell_execute and write_file ask on every call (no
  session/always). - Allow for Session/Always record a directory or host grant; saved grants go
  through a file-only config helper. Bare tool names in allowed_tools are no longer honoured. -
  Tools receive a ToolContext with the approved read roots and hosts; fetch_url can ask about a host
  it meets mid-fetch from its worker. - Tool results are capped when they enter history, on resume
  and when a session is saved.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **config**: Add observe_max_age_days retention setting
  ([`894d45a`](https://github.com/cleanunicorn/drove/commit/894d45ae5747938d89a084421bdfcee3b6f1839e))

Defaults to 30 days; 0 keeps records forever and negative values are rejected. Persisted by
  Config.save and documented in the settings table.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **models**: Name hidden incomplete downloads in drove models list
  ([`d2896e7`](https://github.com/cleanunicorn/drove/commit/d2896e7cf8ec106cd6e85f9a7b2f5784c21000e1))

After #152 an interrupted download vanished from the list, completion and /v1/models with no sign it
  still used disk space or needed finishing. Add ModelStore.incomplete() (one shared scan with
  list()) and print a hint naming the hidden downloads and how to finish or delete them.

Belongs to: #152.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **observe**: Redact credentials, cap bodies and prune old records
  ([`5c941a8`](https://github.com/cleanunicorn/drove/commit/5c941a82c31655e6e48eeb46b0573e56aace293e))

Saved records replace authorization, proxy-authorization, cookie, set-cookie and x-api-key values
  (any case) with [REDACTED], keep at most 256 KiB of text per body with a truncation marker, store
  binary and multipart bodies as a size and content-type placeholder instead of hex, and stay under
  1 MiB of encoded JSON. Files are created exclusively with mode 0600. prune_records deletes record
  files older than a given number of days by modification time.

The TUI literal-rendering fixture moves its markup value off the Authorization header, which is now
  redacted.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Refactoring

- **chat**: Build tool paths from one join rule
  ([`897f2c7`](https://github.com/cleanunicorn/drove/commit/897f2c7b6c7a7e4754dacc0a8f671f15c4490082))

tool_policy.resolve_path (what the policy decides on) and tools._path_arg (what the tool executes
  on) each expanded ~ and joined to cwd. Add tool_policy.join_tool_path; resolve_path resolves it
  and _path_arg uses it unresolved, so error messages still show the path as written.

Belongs to: #154.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Derive the default port from one rule
  ([`8de7c50`](https://github.com/cleanunicorn/drove/commit/8de7c50395f4f82a0a949c61b5c489f8e8da445d))

tool_policy.canonical_origin (the approval scope) and tools._default_port (fetch_url's resolver)
  each wrote 443-for-https-else-80. Add tool_policy.url_port and use it in both, so the approved
  origin and the connection target cannot drift apart.

Belongs to: #154.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Name the tool dialog choice sets once
  ([`12973c8`](https://github.com/cleanunicorn/drove/commit/12973c85ac1e4df3682ab82e6e9635d021c865a3))

The modal's per-call guard and ChatApp._confirm each spelled out the allow/scoped choice sets.
  Define _SCOPED_CHOICES and _ALLOW_CHOICES next to the modal and use them in all three places.

Belongs to: #154.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **cli**: Print-and-exit for ConfigError in one place
  ([`a6eb37e`](https://github.com/cleanunicorn/drove/commit/a6eb37e05e18435d48a84d764d4e8919cee350f3))

The callback, _load_or_exit and `drove config` each wrote the same error(next_step) + Exit(1) block.
  Extract _exit_config_error.

Belongs to: #148.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **config**: Share the init-only settings sources override
  ([`fb63c57`](https://github.com/cleanunicorn/drove/commit/fb63c57c0b9a39908ef9c5f4de1837f94e50762b))

_FileLlamaServerDefaults and _FileConfig each repeated the same settings_customise_sources returning
  (init_settings,). Move it to an _InitOnlySources mixin placed first in both bases, so the
  "explicit values only" policy is written once.

Belongs to: #148.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **downloader**: Import the download record's temp name
  ([`2b3469c`](https://github.com/cleanunicorn/drove/commit/2b3469c2446a594551a9a6269d31cf3b3f6fb75d))

_check_targets rebuilt write_state's private temporary file name to vet it. Export
  MANIFEST_TEMP_NAME from download_state and use it in both, so renaming the temp file cannot
  silently drop it from the safety check.

Belongs to: #152.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **models**: Share the flat-GGUF primary check
  ([`2d4b14a`](https://github.com/cleanunicorn/drove/commit/2d4b14ae7f4e76a245f18a35e66b0707d03c656a))

list() and _find_primary each spelled out "not a projector, a file, and servable" for a top-level
  <name>.gguf. Extract _is_flat_primary so list() keeps offering exactly what resolve() serves.

Belongs to: #152 (combined with #147's projector rule).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **observe**: Name the original-body-size rule once
  ([`d3b1d41`](https://github.com/cleanunicorn/drove/commit/d3b1d419f3c5301ce9ebeba4f4d6933f6da3325f))

ObserveContext.to_record wrote "captured length unless an original size was recorded" twice, with
  stray parentheses. Extract _original_size.

Belongs to: #153.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **proxy**: Read the response content type through observe's helper
  ([`796aae9`](https://github.com/cleanunicorn/drove/commit/796aae9965d0d2431e3a5a11cb16c1a99db54ed7))

_counting_stream re-implemented observe's case-insensitive Content-Type lookup to decide whether to
  capture a body, while observe used its own to decide how to render it. Make it
  observe.content_type and use it in both, so capture and rendering read the header the same way.
  Rename the request-parsing local that would otherwise shadow it.

Belongs to: #153.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **stats**: Keep only the newest speed and TTFT sample
  ([`7456a29`](https://github.com/cleanunicorn/drove/commit/7456a29a67286c8a438193d11d514950635ef10d))

Since #153 moved averages onto running sums, the 1024-entry deques were only read at [-1]; the other
  1023 samples per deque were never used. Keep the last sample instead and drop MAX_SAMPLES. The
  bounded-memory test now checks that no per-request collection is retained.

Belongs to: #153.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>


## v0.13.1 (2026-10-27)

### Bug Fixes

- **cli**: Reject unsafe model names in models commands and downloads
  ([`ebded0b`](https://github.com/cleanunicorn/drove/commit/ebded0ba98b7b4a461c5645af0407be510887e9e))

`models delete`, `info` and `config` validate the name through the shared rule and exit 1 with the
  rule spelled out, so `delete . -y` and `delete .. -y` no longer reach rmtree. `download --name` is
  checked before the models directory is created or the Hub is contacted, an explicit empty --name
  is no longer silently replaced by the inferred name, and DownloadPlan.destination validates the
  (mutable) local name and refuses a directory that resolves outside models_dir for every backend.
  add-tts and the TTS download path reuse the shared validator.

The containment helper becomes public (is_within) now that the downloader uses it. Download fixtures
  that used a name with a space switch to a hyphenated name, since whitespace is no longer valid.

Refs #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **downloader**: Follow inward links when checking download targets (V2)
  ([`ec1119c`](https://github.com/cleanunicorn/drove/commit/ec1119cc012f7ddcd1ac97cb2e4078b168645a32))

An inward symlink (for example .cache -> another folder in models_dir) passed the containment check,
  and the walk never looked behind it, so an outward link further down still let the Hub write
  outside models_dir. Check every target and everything under it by resolved real path, following
  directory links and tracking scanned real directories so link loops terminate.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **downloader**: Refuse download targets that resolve outside models_dir
  ([`146e0f5`](https://github.com/cleanunicorn/drove/commit/146e0f5882b9d867551d3784e61b34a1c750542e))

DownloadPlan only checked its destination directory, so a pre-existing symlink below it (a nested
  repository folder, a weight file, the Hub's .cache folder, or the sidecar config and TTS marker
  written afterwards) could redirect the download's reads and writes outside models_dir. Every one
  of those paths is now checked before local status is read or anything is downloaded; the CLI
  reports the offending path and exits 1, and the quant picker shows "local status unavailable" for
  such a plan.

Refs #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **downloader**: Reject escaping symlinks anywhere under the Hub cache (V2)
  ([`66d9582`](https://github.com/cleanunicorn/drove/commit/66d9582e1e9068de90b6e576dbe51468b8b8e2b4))

The previous check covered only dest/.cache itself, so an outward symlink at .cache/huggingface or a
  metadata/lock/partial file beneath it still let huggingface_hub write outside models_dir. Walk the
  existing cache subtree without following links and refuse the download if any entry escapes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **model-store**: List only model names that resolve accepts
  ([`de0010e`](https://github.com/cleanunicorn/drove/commit/de0010ed92cd668af2f0eb45e828a7a833587200))

list() and complete() still offered on-disk names that break the new naming rule (a flat `my
  model.gguf`, a `bad name/` directory or namespace), so a name copied from `drove models list`,
  completion or GET /v1/models failed with an invalid-name error. Inventory and completion now apply
  the same rule as resolve(); such models stay hidden until renamed on disk, as the docs and
  changelog now say. Completion also skips a flat GGUF whose sidecar escapes, matching list().

The model-list output test keeps its literal-markup check by putting the markup in the models
  directory path instead of a model name.

Refs #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **model-store**: List, complete and resolve the same model candidate (correctness-4)
  ([`f713ea6`](https://github.com/cleanunicorn/drove/commit/f713ea6d9535081e43b1880ece8d1e8c3150236b))

The sidecar fix made resolve() skip an unservable first candidate while list() dropped the whole
  model after checking only candidates[0], and complete() never checked servability at all.
  _add_entry now picks the first servable file in the same format priority as _find_primary, and
  complete() is derived from list() so the three always agree.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **model-store**: Treat a model with an escaping sidecar as absent
  ([`01a694c`](https://github.com/cleanunicorn/drove/commit/01a694ce99c9e66567a23705cc3ffee8f605aed1))

Config reads and writes follow a model's sidecar TOML, so a contained primary with a sidecar
  symlinked (or dangling) outside models_dir let `models config`, `models info`, the server and
  integrations read or overwrite an external file. The store now only serves, lists and resolves a
  primary whose sidecar also stays inside models_dir, and the HuggingFace alias scan skips TOML
  files that resolve outside it.

Deletion is unchanged: it unlinks a sidecar symlink without following it.

Refs #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **model-store**: Validate model names and keep paths inside models_dir
  ([`b7773a7`](https://github.com/cleanunicorn/drove/commit/b7773a78cee86aa6299f5d2112fabfaefef7756a))

Add validate_model_name, the one shared rule for model names (`name` or `org/name` safe-token
  segments, `:QUANT` allowed), and call it before any disk access in ModelStore.resolve and
  find_root. Every candidate path is also resolved and must stay strictly inside models_dir, so
  traversal names and symlinks that point outside the store are treated as absent; list() and
  complete() skip them too. Alias results that are not valid names are ignored.

The two CLI deletion tests that relied on an outward root or namespace symlink being a model now
  assert that deletion refuses it and leaves the link and its target untouched.

Refs #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **observe**: Contain record files and unfiltered scans to observe_dir
  ([`d0bcaaf`](https://github.com/cleanunicorn/drove/commit/d0bcaaf394351f3bce5db549734dd3612fdefda7))

save_record checked the model directory but then wrote through a symlink at the record file itself,
  and listing, search and lookup without a model filter followed directories and record files that
  point outside observe_dir. The record path is now checked before writing, and scans skip
  directories and symlinked record files that resolve outside observe_dir.

Refs #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **observe**: Never write or read records outside observe_dir
  ([`66a3f8b`](https://github.com/cleanunicorn/drove/commit/66a3f8b085c8e0e0564889a8eac2e758473364c4))

save_record validates a record's model through the shared rule (None still maps to `_unknown`) and
  refuses a model directory that resolves outside observe_dir, so a crafted model name or an outward
  namespace symlink writes nothing. A model filter on listing or lookup is treated as a key: an
  invalid or escaping one matches no records instead of reading another directory. The namespaced
  `org/repo:QUANT` layout is unchanged.

Refs #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **proxy**: Reject an explicit null model with 400
  ([`6e9a3e6`](https://github.com/cleanunicorn/drove/commit/6e9a3e6a0c94ce4c99caeb0c478b4232eb51975e))

A JSON `"model": null` was treated like an omitted field and routed to the default model. It is a
  non-string model value, so it is now answered with HTTP 400 before ensure_running, like any other
  invalid model; only an omitted field falls back to the default or loaded model. The speech
  default-routing test's null case now expects the 400, and docs/api.md says so.

Refs #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **proxy**: Reject an invalid model name with 400 before any lookup
  ([`2ffad6a`](https://github.com/cleanunicorn/drove/commit/2ffad6a2e7f6842700d274f1b4b7ba9b74548473))

The proxy now tells an absent `model` field apart from a present one. Any present value that is not
  a valid model name (a non-string, an empty string, a traversal or absolute path, a file-valued
  multipart field) is answered with HTTP 400 before ensure_running, the model store or the disk is
  touched, and the request counters settle. JSON null and an omitted field still route to the
  default or loaded model; a configured default_model is validated the same way.

ServerManager.ensure_running validates on entry too, so non-canonical spellings such as `./qwen`
  never become separate instance keys.

Refs #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **server**: Keep backend output and shutdown from blocking
  ([`e9aec63`](https://github.com/cleanunicorn/drove/commit/e9aec63b8a920fabff614eec10dd50f30c2a57bc))

Backends were spawned with stdout piped but never read. Once ~192 KiB queued (kernel pipe plus
  asyncio's paused buffer), the next write blocked the worker; uvicorn's access log reached that in
  ~2,600 requests, freezing the ASR and TTS workers. stdout now goes to DEVNULL, stderr keeps its
  bounded drain, and both workers turn off the access log.

The wait after SIGKILL was unbounded and ran under the manager lock. asyncio only reports an exit
  once every pipe closes, so a descendant holding stderr hung every later lifecycle call. Bound that
  wait with STOP_TIMEOUT_SECONDS, warn on timeout, and always cancel the stderr reader.

Closes #114

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **server**: Keep draining stderr after a stop gives up
  ([`80b1bed`](https://github.com/cleanunicorn/drove/commit/80b1bed02ce953d72462f68128191993aa637165))

Cancelling the stderr reader once the bounded post-kill wait expired left the pipe of a surviving
  descendant open with no consumer: the descendant blocked after ~64 KiB more output and the
  transport and fd leaked. Leave the reader running; it ends on EOF, after which asyncio releases
  the pipe and subprocess transports itself.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **server**: Kill a backend whose stop is interrupted
  ([`36fbc6f`](https://github.com/cleanunicorn/drove/commit/36fbc6f5e81946cb8b7f81d45d7405efeac79ea9))

The instance is untracked before termination starts, so cancelling a stop during the SIGTERM grace
  wait left a backend that ignores SIGTERM running with no owner. SIGKILL it (its process group for
  TTS) before propagating; the stderr reader drains it until EOF.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **server**: Release a request's slot only on the instance it claimed
  ([`8653fa0`](https://github.com/cleanunicorn/drove/commit/8653fa0f884a3d45ecebd8c1e342aec7c5366ee6))

request_finished looked the instance up by model name, so a request that outlived its instance (for
  example one whose backend died and was replaced by a peer's cold start) decremented the
  replacement's count and could make a busy model look idle and evictable.

ensure_running(claim=True) and request_started now return the claimed instance as an opaque slot,
  the proxy passes it back to request_finished, and a slot that is no longer the registered instance
  releases nothing. Callers that pass no slot keep the name-based behavior.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **server**: Reuse a peer's cold start after an eviction drain
  ([`3ab9aca`](https://github.com/cleanunicorn/drove/commit/3ab9aca474eda0b68f6b4b89645c5c520d3b0118))

Two requests for the same unloaded model could both wait for a busy model to drain with the manager
  lock released. Whichever reacquired it second evicted another busy model (or drained the first
  request's new instance) and started a second process, overwriting the first one's registry entry
  and orphaning it.

Eviction now returns as soon as the target model is running, and ensure_running rechecks the
  registry after eviction to claim the peer's instance. _start refuses to overwrite a tracked
  instance or idle watcher, so no process can be left untracked.

Closes #113

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **sessions**: Keep chat session files inside sessions_dir
  ([`48b8aee`](https://github.com/cleanunicorn/drove/commit/48b8aee7ceffdaff45eda99e71c3a059620e4957))

Session storage joined the chat model label onto sessions_dir verbatim, and that label can come from
  a remote endpoint's /v1/models catalog, so an id such as `../../x` or an absolute path wrote
  session JSON anywhere the user can write. Valid model names keep their `org/name` layout; every
  other label (or one whose directory resolves outside through a symlink) maps to a flat
  `_<name>-<digest>` folder that no valid name can collide with. Switching models now reports such a
  save failure instead of raising.

Refs #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Documentation

- Document the model naming rule and the 400 for invalid names
  ([`e9a1306`](https://github.com/cleanunicorn/drove/commit/e9a13068ee3d23c1ffc6b1b7a1522341bd145d68))

Describe the `name` / `org/name` rule and the models_dir containment (and its symlink consequence)
  next to the naming guidance, note in the API reference that an invalid `model` is answered with
  HTTP 400, and add the Security entry under [Unreleased].

Closes #110

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **changelog**: Consolidate combined unreleased entries
  ([`f26f375`](https://github.com/cleanunicorn/drove/commit/f26f375e8e0dba0fbcb6f8a5b507049fd6a587b9))


## v0.13.0 (2026-10-26)

### Bug Fixes

- **chat**: Show a tool that crashes while running as failed
  ([`ab087ba`](https://github.com/cleanunicorn/drove/commit/ab087ba202396906ee2fb908411103d094b244be))

A tool that raised after it started (for example read_file with a null path) was relabelled "not
  run", which is reserved for calls a stop prevented. Finish the entry as failed with the error
  text, then let the existing error handling end the turn.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Documentation

- **chat**: Document live tool feedback and the context meter
  ([`3446ea5`](https://github.com/cleanunicorn/drove/commit/3446ea5ff484a77639c3ab0f55ba124fe9c19acd))

Describe the running tool entry and its outcomes, streamed shell output, the header context meter
  with its fallbacks and resets, and the new drove.context_length catalog field; add the Unreleased
  changelog entries.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Features

- **chat**: Add a context-window meter to the header
  ([`21bd876`](https://github.com/cleanunicorn/drove/commit/21bd8761c2711d5d48fe97572d3305569453d520))

The chat now shows how full the model's context is after each reply: a bar-only meter at the top
  right with used / total tokens in its tooltip and on the Info screen, turning to the warning
  colour from 80%. The size comes from drove.context_length on /v1/models; when it is unknown the
  header shows the raw token count instead of a guessed fill, and with no usage the meter is hidden.
  Usage now belongs to the latest response, stays visible while the next reply streams, and resets
  on /new, model switch, resume, stop or error.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Advertise known context length on /v1/models
  ([`f013c29`](https://github.com/cleanunicorn/drove/commit/f013c29ed370534bf1aad9aa23843b2c0a30a32e))

The chat needs the active model's context size to draw a fill meter, but ModelSpec.context_window
  falls back to a guessed 4096. Report the size only when a sidecar, extra_args or GGUF metadata
  establishes it, as drove.context_length (null otherwise), and parse it strictly on the client.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **chat**: Show running tools live with streamed shell output
  ([`6714556`](https://github.com/cleanunicorn/drove/commit/6714556947d49037345ae0e286a25d1f1084bfbb))

A tool call now appears as soon as it starts, names the tool and its arguments, and updates in place
  to done, failed, denied or not run. The activity line names the running tool with elapsed seconds,
  and shell output streams into the entry through a bounded tail that a 10 Hz timer drains, so the
  worker thread never touches widgets. Stop keeps waiting for the tool and records its real result,
  as before.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

- **tools**: Stream shell output and mark failed tool results
  ([`385764e`](https://github.com/cleanunicorn/drove/commit/385764eb1d24c940ee584afbfbea8656ca68afb6))

The chat needs live shell output and a reliable failed/done distinction. Foreground shell_execute
  now drains stdout and stderr in raw chunks with one deadline and reports them to an optional
  callback; the result text sent to the model is unchanged. Errors, timeouts and nonzero exits come
  back as ToolFailure, a str subclass, so the UI never parses output to guess.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Refactoring

- **chat**: Drop the unused tool-entry state tuple
  ([`7875cfb`](https://github.com/cleanunicorn/drove/commit/7875cfb6cec42d8fc45962048d9f317a41d061f6))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>


## v0.12.0 (2026-10-25)

### Bug Fixes

- **chat**: Save tip progress without persisting env overrides
  ([`5ab02de`](https://github.com/cleanunicorn/drove/commit/5ab02de4bad2d1c7dbb33d3624277be3a2640c54))

Tip progress was saved via load_config()+Config.save(), which writes the fully resolved config —
  DROVE_* env overrides and every default — into the TOML file on the first Enter press. Persist
  only tui_completed_hints instead.

Also request stream_options.include_usage so llama-server actually reports token usage for Info,
  drop an unused "compact" CSS class, and fix CHANGELOG bullet spacing.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Documentation

- Update installation and require temporary worktrees
  ([`ecbb3ce`](https://github.com/cleanunicorn/drove/commit/ecbb3cef6f547484cff0f2a4933ef1a2bbcd8e84))

### Features

- **chat**: Reclaim conversation space with progressive guidance
  ([`b7183bc`](https://github.com/cleanunicorn/drove/commit/b7183bc84e698124f4822d77d2cd6d528a2d7a34))


## v0.11.0 (2026-10-25)


## v0.10.0 (2026-10-25)

### Bug Fixes

- **chat**: Exclude unservable weights from model selection
  ([`61b0ee3`](https://github.com/cleanunicorn/drove/commit/61b0ee324a1e29b638556f62398d9f0eea7f571c))

Resolve catalog entries through ModelStore before advertising chat capabilities. Keep unsupported
  weights visible in inventory with unknown capabilities.

Add six regression cases covering safetensors, bin and pt in local and namespaced directories,
  including availability after conversion to GGUF. All six failed before the fix; make fmt, make
  lint, make typecheck and make test pass (491 tests).

Sweep: 2 inventory-to-capability paths found; 1 fixed in the reviewed catalog change; 1 left in
  pre-existing bulk integration registration outside the review target.

- **chat**: Follow buffered markdown layout while streaming
  ([`5ac1bb3`](https://github.com/cleanunicorn/drove/commit/5ac1bb332f38ef14cb5de429730cd5286dab0198))

- **chat**: Merge literal rendering from PR #127
  ([`8bdb761`](https://github.com/cleanunicorn/drove/commit/8bdb761b1b77ceccd3864fcb032c78133bdcd142))

# Conflicts: #	CHANGELOG.md #	docs/cli.md #	tests/test_tui.py

- **chat**: Render messages and tool approvals literally
  ([`6d97219`](https://github.com/cleanunicorn/drove/commit/6d97219cf0297db03b572d07abe168437b4d80a7))

- **chat**: Scope stopped tool results to the current batch
  ([`8108e82`](https://github.com/cleanunicorn/drove/commit/8108e8228a340413173e6050f5186c16bc6ae684))

- **cli**: Normalize deletion preview display paths
  ([`a3550eb`](https://github.com/cleanunicorn/drove/commit/a3550eb43925557db27633ca30e3e37e0549f1b0))

- **cli**: Warn for loaded models beneath deletion targets
  ([`92a92df`](https://github.com/cleanunicorn/drove/commit/92a92df061ebc6e0810cb94a836b2ea497e51f8f))

- **download**: Preserve configuration in recovery commands
  ([`2ab2521`](https://github.com/cleanunicorn/drove/commit/2ab2521d5a27869f014ef60c1080c9b246fb8af4))

Retain the effective config path in Retry, Resume, and repair inspection commands. Replay
  regressions cover failures and interruptions during transfer and configuration, including
  shell-quoted paths.

Resolve merged documentation inconsistencies around default model selection and Markdown chat
  controls. Validation: formatting, lint, strict type checking, 797 tests, and both browser gates
  passed. Independent review verified the recovery correction.

- **download**: Retry configuration after completed transfers
  ([`50c1086`](https://github.com/cleanunicorn/drove/commit/50c1086daa6d82012f0af26493c5e38f55a2f83d))

- **observe**: Merge sanitized Markdown previews from PR #129
  ([`f42164c`](https://github.com/cleanunicorn/drove/commit/f42164cb2042c2197e4dadfee14730fb706cbf3f))

# Conflicts: #	CHANGELOG.md

- **observe**: Sanitize offline Markdown previews
  ([`3be0dbb`](https://github.com/cleanunicorn/drove/commit/3be0dbb1726ef9e90891a6367585975d7e3d3264))

- **observe**: Stop background scans when quitting
  ([`caf8c50`](https://github.com/cleanunicorn/drove/commit/caf8c50edc236bacb24aea884b3b2f51c2f5845a))

- **proxy**: Supply the default model to speech workers
  ([`540e28b`](https://github.com/cleanunicorn/drove/commit/540e28bc816c6d58b9ac1922b95ab3f0f96dd99d))

### Documentation

- Redesign documentation around user tasks
  ([`7d84e00`](https://github.com/cleanunicorn/drove/commit/7d84e00bded189c55ef4db5a1c6bc5e83f3c78c4))

### Features

- **chat**: Add searchable model selection and in-chat switching
  ([`298c683`](https://github.com/cleanunicorn/drove/commit/298c6838a4c58c08965d0e6c9fd1a1570a772e8e))

- **chat**: Merge Markdown replies and actionable errors from PR #130
  ([`0bf2845`](https://github.com/cleanunicorn/drove/commit/0bf28455948ca53550fbf2e57d34b7f89151af3e))

# Conflicts: #	CHANGELOG.md #	docs/cli.md #	docs/text-and-vision.md #	src/drove/tui.py
  #	tests/test_tui.py

- **chat**: Merge reply cancellation and prompt queues from PR #128
  ([`52269b6`](https://github.com/cleanunicorn/drove/commit/52269b6a00aef0f8844e61bd6d4c863d02845f38))

# Conflicts: #	CHANGELOG.md #	docs/cli.md #	src/drove/tui.py #	tests/test_tui.py

- **chat**: Render markdown replies and actionable errors
  ([`e6ddf64`](https://github.com/cleanunicorn/drove/commit/e6ddf64793b5a1020039bf1d226ab70481f1ee3a))

- **chat**: Stop replies and manage queued prompts
  ([`fe9db35`](https://github.com/cleanunicorn/drove/commit/fe9db351e5f161a5e92fa795be6f89e0c539778f))

- **cli**: Merge actionable command feedback from PR #126
  ([`9b32a95`](https://github.com/cleanunicorn/drove/commit/9b32a9517c9752538814337ed576b197416919d3))

# Conflicts: #	src/drove/cli/main.py #	src/drove/cli/models.py #	src/drove/cli/output.py

- **cli**: Merge contextual confirmations from PR #109
  ([`b5a049d`](https://github.com/cleanunicorn/drove/commit/b5a049dbec3adda7396a29c6066310468f041711))

# Conflicts: #	CHANGELOG.md #	docs/cli.md #	src/drove/cli/models.py #	tests/test_cli_models.py
  #	tests/test_cli_output.py

- **cli**: Show contextual model confirmations
  ([`dd65433`](https://github.com/cleanunicorn/drove/commit/dd654333b59b0007d333b8a1fc16934adf8d6d30))

- **cli**: Unify actionable command feedback
  ([`856bca3`](https://github.com/cleanunicorn/drove/commit/856bca30919a83c00a42045aa721235da274b091))

- **config**: Add a default model for server and chat
  ([`eca8be2`](https://github.com/cleanunicorn/drove/commit/eca8be258c92311c0ebdbbe418809ad529a14628))

- **config**: Merge default model support from PR #106
  ([`6597503`](https://github.com/cleanunicorn/drove/commit/6597503ab1624d966e04cdb0a526af8d473dec31))

# Conflicts: #	CHANGELOG.md #	docs/api.md #	docs/cli.md #	docs/text-and-vision.md
  #	src/drove/cli/main.py #	tests/test_cli_chat.py #	tests/test_proxy.py

- **download**: Merge progress and recovery guidance from PR #107
  ([`6a8bc6d`](https://github.com/cleanunicorn/drove/commit/6a8bc6daf2c510643af8c3375bd316531d359b04))

# Conflicts: #	CHANGELOG.md

- **download**: Show unified progress and recovery guidance
  ([`d04e324`](https://github.com/cleanunicorn/drove/commit/d04e324ab59c203d9677971a372e068dad74a44f))

- **models**: Guide quantization selection
  ([`6b7a037`](https://github.com/cleanunicorn/drove/commit/6b7a037a9becdfddc309fbd1dba3b7bea908a702))

- **models**: Merge quantization guidance from PR #108
  ([`58ae951`](https://github.com/cleanunicorn/drove/commit/58ae951f59bd8feeb84af0254b5ea48dea7517ca))

# Conflicts: #	CHANGELOG.md #	docs/cli.md

- **observe**: Add accessible request browsing states
  ([`0780a8f`](https://github.com/cleanunicorn/drove/commit/0780a8fa3cc3392091bfa9d0998cfb133b168ded))

- **observe**: Merge accessible request navigation from PR #131
  ([`0a6a322`](https://github.com/cleanunicorn/drove/commit/0a6a322613b14e5ac257f28668eafdd91a78015e))

# Conflicts: #	CHANGELOG.md #	docs/operations.md #	src/drove/observe_web_static/app.js
  #	tests/test_observe_web.py

- **observe**: Merge responsive terminal paging from PR #132
  ([`d8e0bf5`](https://github.com/cleanunicorn/drove/commit/d8e0bf58f6399c5f902ad3bf8c9187c8b23e6ba4))

# Conflicts: #	CHANGELOG.md

- **observe**: Page terminal logs and keep search responsive
  ([`412eafd`](https://github.com/cleanunicorn/drove/commit/412eafdce846db47ac9e5de29dea92619ebb196b))


## v0.9.0 (2026-10-24)

### Bug Fixes

- **stats**: Capture audio usage in server status
  ([`d3c3702`](https://github.com/cleanunicorn/drove/commit/d3c3702b1447d0e7ab98713cafe50e927cc27782))

### Features

- **tts**: Add local KittenTTS 2 speech generation
  ([`5690ea3`](https://github.com/cleanunicorn/drove/commit/5690ea307327392c0302819171f3ecd1e965520b))


## v0.8.3 (2026-10-17)

### Bug Fixes

- **cli**: Show the bind address when status is network-exposed
  ([`6dadd1f`](https://github.com/cleanunicorn/drove/commit/6dadd1fdd0999c27070ea1b9553b8ec5ad42cc15))

server status connects to 127.0.0.1 when listen_host is 0.0.0.0, so the heading URL hid that the
  server is reachable from the LAN. Show a "listening on ..." line in the default view for any
  non-loopback bind.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>


## v0.8.2 (2026-10-17)

### Bug Fixes

- **cli**: Honor --config and DROVE_CONFIG when saving config
  ([`36ad604`](https://github.com/cleanunicorn/drove/commit/36ad604c88164afe5682403b02709a3248247c39))

drove config KEY VALUE and drove init always saved to ~/.config/drove/config.toml, even with
  --config, overwriting the real file with the values loaded from the --config file. DROVE_CONFIG
  was documented but never read. Resolve the config path once in config.py (--config, then
  DROVE_CONFIG, then the default) and use it everywhere.

Also name text-to-speech models correctly in drove integrate errors, and document text-to-speech
  across the README, getting-started, CLI, configuration, architecture, and agent docs, with
  examples.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>


## v0.8.1 (2026-10-17)

### Bug Fixes

- **cli**: Use click.clear in status --watch
  ([`feb463c`](https://github.com/cleanunicorn/drove/commit/feb463cb41cc815e54a1ae7751e27ba2e57057fb))

typer.clear does not exist in current Typer releases, so --watch crashed with AttributeError on
  terminals.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>


## v0.8.0 (2026-10-17)

### Features

- **cli**: Prioritize concise summaries and optional diagnostics
  ([`f787a10`](https://github.com/cleanunicorn/drove/commit/f787a10fb09ff8d34366d3d13a9338c00339380f))

- **cli**: Redesign inspection output and status reports
  ([`dea5750`](https://github.com/cleanunicorn/drove/commit/dea5750f5797a0f48e89c244e40f31f874cdd106))


## v0.7.0 (2026-10-17)

### Bug Fixes

- **audio**: Address TTS lifecycle, configuration, and voice review issues
  ([`6b5b160`](https://github.com/cleanunicorn/drove/commit/6b5b16083973c097aaf19ed18ce239f0989428cd))

- **audio**: Preserve TTS defaults and clean up cancelled startup
  ([`948c339`](https://github.com/cleanunicorn/drove/commit/948c3397966d98428ed4b0b659ea2e17b7f61165))

### Features

- **audio**: Integrate Cathy engines with the OpenAI speech API
  ([`1916920`](https://github.com/cleanunicorn/drove/commit/1916920aeaec9d5e5d73cce0cc755ca818851f15))

Add managed, isolated Python 3.12 workers for Kokoro, Qwen, Chatterbox, and Fish, with OpenAI speech
  requests and binary audio responses.

Validation: 19 worker tests passed; proxy and model lifecycle smoke checks passed using Python 3.12
  with syntax compatibility adjustments in a temporary copy. Full format/lint/typecheck/test gates
  and real synthesis remain unverified because the sandbox blocks Python 3.14 and model/dependency
  downloads.

- **models**: Integrate TTS downloads with model management
  ([`6efed55`](https://github.com/cleanunicorn/drove/commit/6efed55132f32e973ade1c32673be3df7e3ca6b5))

Recognize Cathy engine repositories in models download, preserve nested voice and tokenizer files,
  resume downloads, and configure workers to load local checkpoints. Support compatible repositories
  through --tts-engine.

Validate with make fmt, make lint, make typecheck, and make test (361 passed), plus the 19 speech
  worker tests under Python 3.12 and live metadata resolution for all four canonical TTS
  repositories.


## v0.6.0 (2026-10-13)

### Documentation

- Adopt the agents-library AGENTS.md template
  ([`24811cf`](https://github.com/cleanunicorn/drove/commit/24811cf698b698b44d4f29551e4ce1e4abfd7f82))

Rewrite AGENTS.md to follow the agents-library agent-guide template. All content from the old
  Claude-Code-style guide is preserved and reorganized: the make-target commands (and the
  always-use-make rule with the single-test uv fallback), the architecture map (config.py,
  model_config.py, model_store.py, server_manager.py, backend.py, workers/asr.py, proxy.py, cli/),
  the request flow, the global and per-model config formats, the systemd service targets, and the
  CHANGELOG policy. New: verified prerequisites, golden rules, the GitHub flow, PR title rules (as
  enforced by pr-title.yml), and a release-automation section matching release.yml, pyproject.toml
  [tool.semantic_release], and docs/deploy.md.

CHANGELOG.md gains the required [Unreleased] entry.

- **architect**: Journal — verify claims against pinned toolchain
  ([`06a37fb`](https://github.com/cleanunicorn/drove/commit/06a37fb27c0faf30df5beef2c799fbe3828b488d))

Records a correction made on PR #79's motivating evidence (an except-clause pattern claimed as a
  Python 3 SyntaxError is actually valid under this repo's pinned Python 3.14 via PEP 758), and the
  decision not to open a third unreviewed Architect proposal while #77 and #79 are still pending
  review.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01EaHRt5WStkkmPGTacGwhvA

- **architect**: Propose backend adapter extraction for ServerManager
  ([`534664b`](https://github.com/cleanunicorn/drove/commit/534664b766bbd6f27d8e575219953e04815d28c5))

Structural-debt review: ServerManager branches on backend type (llama vs ASR) inline for command
  building and prompt-cache eligibility, while backend.py only handles detection. Recent llama-only
  feature work (prompt cache persistence, cache flags) grew this special-casing further.
  Proposal-only doc, zero functional code changes.

- **architect**: Propose shared SSE delta-parsing module
  ([`01ac071`](https://github.com/cleanunicorn/drove/commit/01ac071702da55d36ffd7780b1c5f18aded03484))

Structural-debt review: observe.py, proxy.py, and tui.py each independently reimplement parsing of
  llama-server's `data: {...}` SSE chat-completions stream (line filtering, JSON decode,
  choices[].delta walk, tool-call accumulation by index). The three implementations have already
  drifted (proxy.py/observe.py require "data: " with a space, tui.py accepts "data:" without one),
  and proxy.py's copy has no direct unit test coverage today. Proposal-only doc, zero functional
  code changes.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01CvQEdCs7mwjEVyCLzzCtrd

- **readme**: Describe drove integrate's final behavior
  ([`0b22ce3`](https://github.com/cleanunicorn/drove/commit/0b22ce37f0200f2095072ec3fab088408eaabda2))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

### Features

- **integrate**: Register drove models with pi, opencode and Kilo Code
  ([`8e951c4`](https://github.com/cleanunicorn/drove/commit/8e951c453f9d32da689923dfba27261bdd69d4e8))

Add `drove integrate <agent> [MODEL...]`, which writes a `drove` provider into a local coding
  agent's config with each model fully set up: the context one request can use (from ctx_size,
  extra_args, slot layout and the GGUF training context), an output limit, vision, tool calling and
  reasoning support read from the effective chat template.

- pi: ~/.pi/agent/models.json, default model in the settings.json beside it - opencode:
  ~/.config/opencode/opencode.json[c] - Kilo Code CLI: ~/.config/kilo/kilo.json[c] (and
  opencode/config.json names)

Merges keep other providers and hand-added keys at any depth. A full sync prunes models drove no
  longer has and clears selections that pointed at them. Writes are all-or-nothing across files,
  keep .bak copies, preserve permissions and symlinks, and roll back on failure. Supports --default,
  --remove, --dry-run, --file, --base-url and --provider-id.

The GGUF reader now also exposes architecture, name, training context length and chat template.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>


## v0.5.0 (2026-08-18)

### Documentation

- Redesign README and refresh getting-started guide
  ([`b00599a`](https://github.com/cleanunicorn/drove/commit/b00599a99e2ec35ff6e76251253fd06481b4f876))

Rework the README into a modern, scannable landing page and expand the getting-started guide to
  match. Both now document the full feature surface, including capabilities that were previously
  undocumented.

README: - Themed logo header, nav bar, "Why drove?" framing, 60-second quickstart - Feature table
  and redrawn lazy-lifecycle diagram - Collapsible guides for text, vision, speech-to-text, chat,
  observability, status, model management, and configuration - Complete command reference table and
  HTTP endpoint list

Getting started: - Prerequisites section (uv, llama-server, ffmpeg) - Three ways to send a first
  request (chat, OpenAI SDK, curl) - New sections: speech-to-text, vision, model management,
  terminal chat, server status, and observability - Cross-links to the other docs

Docs-only; no code changes.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

### Features

- **config**: Expose llama-server cache flags and an extra_args passthrough
  ([`f9e1c96`](https://github.com/cleanunicorn/drove/commit/f9e1c965d01682a0c6df8f949f23a94f7dd86fb6))

ModelConfig was a closed allowlist, so the flags that control prompt caching could not be set at all
  — a sidecar key that is not declared is dropped silently.

Adds cache_prompt, cache_reuse, cache_ram, ctx_checkpoints, context_shift, swa_full and load_mode,
  plus an extra_args list that is appended verbatim after every generated argument (llama-server
  takes the last occurrence of a flag, so these win). Booleans that llama-server exposes as a
  --flag/--no-flag pair now emit the negated form when set to false instead of disappearing.

Also fixes n_parallel, which emitted --n-parallel: llama-server rejects that outright ("error:
  invalid argument: --n-parallel") and the model never started. It now maps to --parallel.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

- **server**: Keep the prompt cache across model sleep and wake
  ([`dd30f38`](https://github.com/cleanunicorn/drove/commit/dd30f38595f051bd22b41ef8f35f66293e5963fd))

llama-server keeps its prompt (KV) cache in RAM, and drove kills that process on every idle timeout
  — so the first request after a model wakes re-processed the whole prompt.

drove now saves each idle slot's KV cache through /slots?action=save before stopping a model and
  restores it once the next process is healthy. Cache files older than prompt_cache_ttl_seconds
  (default 1 hour) are deleted when drove starts and before each model start, which is what keeps
  the directory bounded. On by default; set prompt_cache = false to opt out.

Measured end to end on gemma-4-E4B, 20-token prompt: a cold request processes 20 tokens, and after
  an idle shutdown and wake the same request processes 1 and reads 19 from cache.

A restored cache is only reusable if the whole cache was serialized, so sliding-window models need
  --swa-full — without it the restore reports success and yields zero reuse. That flag costs memory
  (5.8 GB -> 7.0 GB at 32k context on gemma-4-E4B), so drove reads the GGUF header and adds it only
  for models that declare a sliding window. A new minimal GGUF reader answers that question without
  pulling in the gguf package; the answer is cached per file revision and read off the event loop,
  since proving a model has no window means walking the tokenizer vocabulary.

Also fixes _stop_instance cancelling the idle-watcher task that calls it. That was harmless while
  SIGTERM was sent before the first await, but with a save step in front of it the CancelledError
  fired first and left the llama-server process running — an orphaned model drove believed it had
  stopped.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>


## v0.4.0 (2026-07-21)

### Documentation

- Clarify make install is not required for development
  ([`0e01446`](https://github.com/cleanunicorn/drove/commit/0e01446010cf86eb96fbb95f1fad941661856d53))

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01M33GHdU7ceBeknt8GxNo4F

- Use make commands in AGENTS.md development section
  ([`58b4f2f`](https://github.com/cleanunicorn/drove/commit/58b4f2f1a47fa5f86aca850ebd0755d16ffe87ca))

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

Claude-Session: https://claude.ai/code/session_01M33GHdU7ceBeknt8GxNo4F

### Features

- Add max_memory budget for loaded models
  ([`a0cf71b`](https://github.com/cleanunicorn/drove/commit/a0cf71bed501049e107effb0b607bb5c4f88b8ae))

Add a max_memory config option (default "0" = unlimited) that caps the combined memory estimate of
  all loaded models, complementing the existing max_loaded_models count limit. Sizes accept decimal
  ("24GB"), binary ("16GiB"), or plain-byte values.

Model memory is estimated from on-disk file size: all shards for sharded GGUF models, all .onnx
  files for ASR models. Eviction keeps the existing idle-LRU-first policy, now looping since memory
  pressure can require evicting several models. A model whose estimate alone exceeds the budget is
  still started (after evicting everything else) rather than refused. The estimate is exposed in
  status as est_memory_bytes.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>


## v0.3.1 (2026-06-19)

### Performance Improvements

- **observe**: Direct record detail lookup and memory-bounded search
  ([`4111faa`](https://github.com/cleanunicorn/drove/commit/4111faaf69fc46667e794cee121cda2739e9005d))

The record detail endpoint parsed every record on disk to find one id, even though record ids double
  as filename stems — resolve the path directly instead. Search loaded and summarized the full log
  in memory before paginating — scan lazily and keep only the requested page.

Refs #59

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>


## v0.3.0 (2026-06-13)

### Bug Fixes

- **server**: Re-evaluate eviction victim after draining
  ([`c6dedf5`](https://github.com/cleanunicorn/drove/commit/c6dedf54e3a6a670f5dac6fdb02bccd64ad05282))

When every loaded model is busy, _evict_if_needed releases the lock to wait for the LRU model's
  in-flight requests to drain. The victim could be re-claimed by a new request (or capacity could
  free up) during that window, yet the old code stopped the stale victim unconditionally on
  re-acquiring the lock.

Loop the selection instead: after each drain, re-acquire the lock and re-evaluate from scratch. An
  idle model is now stopped only if it is still idle, and nothing is stopped if a slot freed up
  while waiting.

Add regression tests for the re-claim and capacity-freed races.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

### Build System

- Consolidate installation into make install and remove install.sh
  ([`0f81870`](https://github.com/cleanunicorn/drove/commit/0f81870519a8af19aa5b9f13c826404127c0a3ea))

There were two install entry points: install.sh (end-user installer from GitHub) and make install
  (dev install from checkout). Fold all of install.sh's behavior — OS check, uv bootstrap, Python
  pin, extras, PATH warning, llama-server warning, quick-start hints — into the Makefile install
  target, delete the script, and update the README, getting-started, deploy docs, and the CI
  smoke-test workflow.

Co-authored-by: Claude Fable 5 <noreply@anthropic.com>

### Features

- **server**: Evict idle models before busy ones at capacity
  ([`94dbe02`](https://github.com/cleanunicorn/drove/commit/94dbe021b7852cc5b22065ed2386d24fd6e46f4a))

When max_loaded_models is reached and a new model must start, prefer evicting a model with no active
  connections instead of the strict LRU. Models still serving in-flight requests are left running;
  the least-recently-used idle model is evicted. Only when every loaded model is busy do we fall
  back to draining the LRU model.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>


## v0.2.0 (2026-06-12)

### Bug Fixes

- **asr**: Cap transcription upload size at 100 MB
  ([`391460b`](https://github.com/cleanunicorn/drove/commit/391460b1b7f208e1a7edcde9ae7b73dab1627ae1))

The endpoint read the whole upload into memory unbounded; drove listens on the network by default,
  so an oversized body could exhaust worker memory. Reject anything past 100 MB with HTTP 413.
  (round-2 review finding security-1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **asr**: Keep ffmpeg error output out of client responses
  ([`e028c87`](https://github.com/cleanunicorn/drove/commit/e028c87bdb131b64b4f242c4b41c2c144d5a09f7))

The decode-failure detail echoed ffmpeg's stderr tail, which can name local paths and tool
  internals; log it server-side and return a generic message instead. (review finding: sanitize
  ffmpeg stderr)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **asr**: Treat WAV files with bogus chunk sizes as non-conforming
  ([`bbec6ee`](https://github.com/cleanunicorn/drove/commit/bbec6eeec8dc83bdc4d5b70ea25cb183e012eaf1))

The chunk reader behind the wave module raises a bare RuntimeError when a chunk size seeks past the
  end of the data, which escaped _read_wav and surfaced as an internal error instead of falling back
  to ffmpeg. Catch it alongside wave.Error and EOFError. (found while writing testing-3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

### Documentation

- **architecture**: Reflect backend-per-model design in architecture.md
  ([`4dcea70`](https://github.com/cleanunicorn/drove/commit/4dcea70ff874bdc7a1a633f214ebfe8c1cc85ac4))

The request-flow diagram and module list still described llama-server as the only backend; ONNX
  speech-to-text models now run the built-in ASR worker. (review finding docs-1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **changelog**: File the selection-echo entry under Added
  ([`839e398`](https://github.com/cleanunicorn/drove/commit/839e398acca4791d03de316cecba5c0e3c86f0ed))

The entry for the feat(cli) echo commit sat under ### Changed; AGENTS.md requires changelog entries
  consistent with the commit type. (round-3 review finding conventions-1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **cli**: Document ONNX quant refs and the CAPS column
  ([`9ded1fd`](https://github.com/cleanunicorn/drove/commit/9ded1fdc0a4aa95cd242f7cb0c0a82cc521488a5))

The download help only showed GGUF-style :Q4_K_M references; ONNX repos take :int8-style variants
  and prompt when several exist. Also explain the vision/stt tags in models list output. (round-2
  review findings docs-1, docs-2, docs-3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **readme**: Present text generation and speech-to-text side by side
  ([`e62e183`](https://github.com/cleanunicorn/drove/commit/e62e1835094a14b34657cc6f3ecca45693b9187e))

Broaden the tagline from "local LLMs" to "local models", add a Text generation section (Gemma 3 12B
  GGUF example with a curl chat request) and a Speech-to-text section (Parakeet ONNX example with a
  curl transcription request and real output), and add a speech-to-text row to the comparison table.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

### Features

- **asr**: Serve speech-to-text models via a built-in ONNX worker
  ([`433e1de`](https://github.com/cleanunicorn/drove/commit/433e1de689b8d9bc0493447daa2a3b1fdaa779ae))

Add first-class speech-to-text support (e.g. NVIDIA Parakeet TDT) managed exactly like LLMs — lazy
  start on first request, idle shutdown, LRU eviction, same listen port:

- New built-in worker (python -m drove.workers.asr) that loads ONNX ASR models via the optional
  onnx-asr package (drove[asr] extra) and serves an OpenAI-compatible POST /v1/audio/transcriptions
  plus /health. Audio is normalized to 16 kHz mono PCM (ffmpeg when available, pure-Python WAV
  fallback). - Per-model backend selection (src/drove/backend.py): .gguf → llama-server, .onnx → ASR
  worker, overridable with the new `backend` config key. New `asr_model` and `asr_quantization` keys
  (excluded from llama args). - ServerManager dispatches the right command per backend; health
  checks and lifecycle are unchanged. - Proxy extracts the model name from multipart form bodies so
  OpenAI audio requests route on the same port as chat completions. - Downloader handles ONNX repos:
  weights + support files (vocab/config), int8 variant selection via org/repo:int8; CLI
  auto-configures asr_model for known repos and shows an `stt` capability tag. - ModelStore resolves
  ONNX model dirs (GGUF still wins in mixed dirs); list/info/delete/config work uniformly.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **cli**: Confirm the quantization selection with an echo
  ([`8746f50`](https://github.com/cleanunicorn/drove/commit/8746f50a4bec8e6a48ebe68efbbf1a07ddc88f88))

Choosing 'default' or 'all' leaves the model name unchanged, so nothing visibly acknowledged the
  menu choice before the plan printed. (round-2 review finding ux-polish-1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **cli**: Prompt for ONNX quantization variants on download
  ([`0a6c907`](https://github.com/cleanunicorn/drove/commit/0a6c907ae96e324a6a808161ec1c63b2f17e1331))

available_quants() only recognises GGUF-style tags and resolve_download had already narrowed the
  plan to the unquantized variant, so ONNX repos with int8 variants never showed the quant choice
  menu. Keep the full ONNX file set on the plan, summarise its variants, and reuse the existing
  prompt. (review finding ux-polish-1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **config**: Validate drove-specific keys when set via the CLI
  ([`2f59793`](https://github.com/cleanunicorn/drove/commit/2f597936e68d636ea7f1bb12f43f2079089c9d3e))

backend is checked against VALID_BACKENDS, and asr_model / asr_quantization must be safe tokens
  (they end up on the ASR worker's command line). Shape validation rather than a whitelist: setting
  asr_model manually is the documented escape hatch for models outside the known-repo table. (review
  finding: validate asr config values)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **install**: Include speech-to-text support by default
  ([`2b3151c`](https://github.com/cleanunicorn/drove/commit/2b3151c106e8840427bfd7b64d6f72e534dbb3c0))

A model that downloads fine but 503s at request time is a confusing default, so make install and
  install.sh now install drove with the asr extra out of the box (~40 MB of CPU ONNX wheels).
  install.sh accepts DROVE_EXTRAS="" for a minimal text-generation-only install, and the plain pip
  extra remains for manual installs. The installer quick-start now mentions downloading a
  speech-to-text model.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

### Refactoring

- **cli**: Extract shared quant-selection helpers
  ([`5f26483`](https://github.com/cleanunicorn/drove/commit/5f264833ed75f4543ff3fb4a274a8a271489df21))

The ONNX and GGUF prompt branches in download_model duplicated the filter-narrow-rename sequence;
  move selection into _select_quant_variant with a shared _apply_quant_selection step. No behavior
  change. (round-2 review finding refactor-2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **downloader**: Check explicit quant first in filter_onnx_quant
  ([`3b68009`](https://github.com/cleanunicorn/drove/commit/3b68009ceefbbbbca9f26979a8276855f6852ce4))

The quant-is-None branch led with inverted logic (filter OUT quantized files); leading with the
  positive case reads in the order the docstring describes. No behavior change. (review finding
  refactor-1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

- **downloader**: Extract _is_onnx_file helper
  ([`7d036be`](https://github.com/cleanunicorn/drove/commit/7d036be97154036b58a5f6c8efc390729ed37a4f))

The .onnx suffix check appeared in _fetch_files_with_sizes and is_onnx_files; centralize it. No
  behavior change. (review finding: extract _is_onnx_file)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>


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
