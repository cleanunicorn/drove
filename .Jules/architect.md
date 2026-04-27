# Architect's Journal

---

## 2026-04-27 - Extract ModelStore to unify model resolution

**Proposal:** Replace two independent model-path lookup implementations in `server_manager.py` and `cli/models.py` with a single `ModelStore` class in `model_store.py`.

**Why now?** Both modules had their own `_find_model_path` / `_find_model` helpers with subtly different behaviour: the CLI used `rglob()` (recursive) while the server used `iterdir()` (non-recursive). A model whose GGUF sat one directory level below the model root would appear in `vllama models list` but fail to launch — a silent, hard-to-diagnose divergence.

**Tradeoffs:**
- 153 lines of duplicated helper code removed across two files
- Single resolution rule (rglob-based, GGUF-first) shared by CLI and server
- No API or CLI interface changes; fully backward-compatible
- Small surface area: one new 110-line file, no new dependencies

**Migration path:** Four steps — create `ModelStore`, replace `server_manager` methods, replace `cli/models` helpers, delete dead code. All in one PR, no feature flag needed.

**Lesson:** Review caught two real bugs introduced during the extraction:
1. The unified `_canonical()` ran alias lookup *before* checking if the exact local path existed, breaking the server's original exact-path-first semantics (P1). Always match the stricter caller's contract when merging dual implementations.
2. The non-GGUF fallback (`.safetensors`, `.bin`, `.pt`) from the CLI helper leaked into the server path — `llama-server` only accepts GGUF, so this caused confusing startup failures rather than a clean "model not found" error (P2). When unifying code paths, audit what each *caller* can actually consume.
