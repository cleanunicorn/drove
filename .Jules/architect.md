# Architect's journal

Critical learnings only — approved/shipped outcomes, rejections with lasting impact,
cross-proposal patterns, and misunderstood tradeoffs. Read this file first, every run.

## 2026-09-14 - The `except A, B:` "syntax error" is not a bug — check `.python-version` first

**Proposal:** N/A — this is a recurring factual error in *other* proposals' "why now"
evidence, not a proposal of its own.

**Why now?** PR #79 ("Add a CI quality gate") cited `server_manager.py:156`
(`except psutil.NoSuchProcess, psutil.AccessDenied:`) as proof that `drove serve`
"cannot start at all on any Python 3 interpreter," verified with `ast.parse` under a
bare `python3`. That's the *second* time this exact claim has been made — PR #74 made
it first, and a follow-up Architect run already corrected it in a comment on #74
(2026-08-10) before #74 was closed. #79 repeated the original mistake anyway.

**Tradeoffs:** Chasing this "bug" would have been pure waste — there is no bug. The
project requires `python>=3.14` (`pyproject.toml`, `.python-version`), and
[PEP 758](https://peps.python.org/pep-0758/) made parenthesis-free multi-exception
`except` clauses valid syntax in 3.14. Verified again this run: `.venv/bin/python`
(3.14.0rc2, the project's actual pinned interpreter) parses and runs the clause
correctly (`ast.parse` succeeds, produces a `Tuple` exception type identical to
`except (A, B):`; a live `try/except` with that exact clause catches correctly), and
`uv run ruff check src/drove/server_manager.py` reports "All checks passed!" — no
E999. The same idiom appears consistently in 11 places across the codebase
(`proxy.py`, `observe.py`, `observe_tui.py`, `cli/main.py`, `tui.py`,
`server_manager.py`), which is itself evidence it's a deliberate, working style choice
under the pinned toolchain, not a typo.

**Migration path:** N/A.

**Lesson:** Never validate a language-syntax claim against the sandbox's default
`python3` — always check `requires-python` / `.python-version` first and reproduce
with that exact interpreter (`uv run python -c "..."` or `.venv/bin/python`). A claim
already corrected once in a PR comment can still resurface in a *later, independent*
run that never reads PR history — which is the whole reason this journal file exists.
Posted the same correction again on #79 (2026-09-14) and linked back to the #74
correction so the pattern is visible in-repo, not just in this file.

## 2026-09-14 - CI-quality-gate proposed twice, still unmerged both times

**Proposal:** N/A — pattern note, not a new proposal.

**Why now?** The "add `.github/workflows/ci.yml` running lint/typecheck/test" idea has
now been an Architect proposal twice: PR #74 (2026-07-27, closed without merging, no
rejection reason recorded by the owner) and PR #79 (2026-09-07, still open/draft,
unreviewed a week later). The underlying idea is sound and not in dispute — it just
hasn't been actioned.

**Tradeoffs:** Re-proposing an idea that's already sitting in an open, unreviewed PR
adds queue noise without adding leverage. Correcting/maintaining the existing PR (see
entry above) is more valuable than opening a third copy.

**Migration path:** N/A.

**Lesson:** Before drafting a new proposal, list open **and recently-closed** PRs
titled `🏗️ Architect: ...` first. If the idea already has an open PR, engage with that
PR (correct it, comment, or leave it) instead of duplicating. If it was closed without
a recorded reason, that itself may be worth asking the owner about rather than
re-proposing blind.

## 2026-09-14 - Backend-adapter proposal (#77) already owns "decompose ServerManager"

**Proposal:** N/A — scope note for future runs.

**Why now?** This run's OBSERVE pass independently flagged `server_manager.py` (863
lines, the largest module) as doing too much — process lifecycle, LRU/memory eviction,
config-change watching, backend command-building, *and* prompt-cache persistence — and
was about to propose extracting the prompt-cache block (`_prepare_prompt_cache_dir`,
`_prune_expired`, `_save_prompt_cache`, `_restore_prompt_cache`, ~180 lines) into its
own module. Checking open PRs first turned up #77 (2026-08-24, still open/draft),
which already proposes extracting *all* backend-specific branching — including the
prompt-cache-eligibility logic — into `LlamaBackend`/`AsrBackend` adapters. That
proposal's scope subsumes the one this run almost opened.

**Tradeoffs:** A second, narrower `ServerManager`-decomposition proposal would compete
with #77 for the same review attention and could conflict with it at merge time (both
touch `_start`, `_save_prompt_cache`, `_prepare_prompt_cache_dir`). Not worth opening.

**Migration path:** N/A — deferred to #77. If #77 is rejected or stalls for a long
time, the prompt-cache-only extraction (smaller, ~180 lines, already has its own test
file `tests/test_prompt_cache.py`) is a reasonable fallback next candidate — scope it
as *prompt-cache module only*, explicitly deferring the backend-command-building half
that #77 also covers, to avoid re-overlapping.

**Lesson:** Same as above — check the open-PR list before designing, not after.
`server_manager.py`'s size is going to keep surfacing as "the" structural debt item
in this repo until one of the pending decomposition proposals actually lands; the
fix here is to get an existing proposal reviewed, not to keep generating new cuts of
the same file.
