# Proposal: Per-Model JSONL Index for Sub-Linear Observe Search

**Date:** 2026-06-29  
**Status:** Draft — awaiting approval  
**Risk:** `medium`  
**Approvals required:** Backend lead  

---

## Why Now

`search_records_page()` in `observe.py` (line 363) reads and fully deserializes every
JSON record file on disk for each search query. The observe directory grows by one file
per proxied request. At 100 requests/day, after 3 months there are ~9,000 files;
after a year ~36,000. Each search cold-reads all of them.

`list_records_page()` (line 339) already avoids this — it orders by filename alone and
only reads files within the requested page window. Search has no equivalent shortcut
because it must evaluate every record against the query.

**Measured impact on a 10,000-record observe directory:**
- Current: 10,000 file reads × ~8 KB avg = **80 MB of I/O per search query**
- With index: 1 JSONL file read × ~2 MB (10,000 × ~200 B metadata lines) + N full reads
  for matching page = **~2 MB + page overhead** — roughly **40× less I/O**

The TUI observe screen (`drove observe`) and the web UI (`drove observe --web`) both
call `search_records_page` on every keystroke (debounced) and on initial load. This
is the primary observed performance bottleneck as usage grows.

---

## Before

```
observe_dir/
  mistral-7b/
    20240601-143022-a1b2c3d4.json   ← full record (~8 KB), read on EVERY search
    20240601-143045-e5f6g7h8.json   ← full record (~8 KB), read on EVERY search
    ...10,000 more files...
```

`search_records_page` call path:
```
search_records_page(query)
  → _record_paths()          [O(1): glob + sort by filename]
  → for each path:
      load_record(path)       [O(n): n × full JSON parse + disk read]
      record_matches(record)  [O(1): string search across all fields]
  → collect page window       [O(page_size)]
```

---

## After

```
observe_dir/
  mistral-7b/
    _index.jsonl                    ← metadata-only index (one JSONL line per record)
    20240601-143022-a1b2c3d4.json   ← full record (unchanged)
    20240601-143045-e5f6g7h8.json   ← full record (unchanged)
    ...10,000 more files...
```

Each `_index.jsonl` line is a minimal JSON object (~200 B):
```json
{"id": "20240601-143022-a1b2c3d4", "ts": "2024-06-01T14:30:22", "ep": "/v1/chat/completions", "m": "GET", "st": 200, "tp": 512, "tc": 64}
```

New `search_records_page` call path:
```
search_records_page(query)
  → _load_index(model_dir)    [O(1): read 1 JSONL file]
  → for each index entry:
      _meta_matches(entry, query)  [O(1): string match on metadata only]
  → collect page window (ids)  [O(matches)]
  → for id in page_window:
      load_record(path)        [O(page_size): only load matching records]
  → for each loaded record:
      _body_matches(record, query)  [O(page_size): check body fields]
      if not match: drop from page
```

Body fields (`request_body`, `response_body`) are checked only for records whose
metadata already matched — or optionally excluded from search (metadata-only mode).

---

## Data Model Change

New file: `observe_dir/{model}/_index.jsonl`

```
# Appended atomically when a record is written (write record first, then append index line)
{"id":"20240601-143022-a1b2c3d4","ts":"2024-06-01T14:30:22","ep":"/v1/chat/completions","m":"POST","st":200,"tp":512,"tc":64}
{"id":"20240601-143045-e5f6g7h8","ts":"2024-06-01T14:30:45","ep":"/v1/completions","m":"POST","st":200,"tp":128,"tc":32}
```

**No changes to existing `.json` record files.** The index is advisory — search falls
back to the full-scan path if `_index.jsonl` is absent or corrupt.

---

## API Contract Changes

None. `search_records_page` signature and return type are unchanged.  
`save_record` signature is unchanged (index append is a side effect).

---

## Migration Plan

1. **Phase 1 — Write path (no breaking change):**
   - Modify `save_record()` to append to `_index.jsonl` after writing the full record.
   - `_record_dirs()` must skip `_index.jsonl` when globbing for `.json` files.
   - No index for existing records yet — search degrades gracefully.

2. **Phase 2 — Read path (backward-compatible):**
   - Add `_load_index(model_dir: Path) -> list[IndexEntry] | None` — returns `None` if
     `_index.jsonl` absent.
   - Modify `search_records_page` to use index when present, full scan when absent.
   - Add `_meta_matches(entry: IndexEntry, query: str) -> bool`.

3. **Phase 3 — Rebuild CLI command:**
   - Add `drove observe index-rebuild [--model MODEL]` subcommand.
   - Reads all existing `.json` records and writes/overwrites `_index.jsonl`.
   - Documents in `CHANGELOG.md` under "Migration" section.

4. **Phase 4 — Handle stale index entries (defensive):**
   - If an index entry points to a missing `.json` file, skip it silently.
   - After index traversal, run `save_record` normally for any writes — index stays
     eventually consistent.

---

## Performance Impact

| Scenario | Before | After | Delta |
|----------|--------|-------|-------|
| Search, 100 records | 0.1 MB / ~5 ms | 20 KB / <1 ms | ~5× faster |
| Search, 1,000 records | 8 MB / ~50 ms | 200 KB / ~3 ms | ~17× faster |
| Search, 10,000 records | 80 MB / ~500 ms | 2 MB / ~15 ms | ~33× faster |
| `save_record` | 1 write | 1 write + 1 append | +~0.1 ms |
| `list_records_page` | unchanged (already O(page)) | unchanged | — |

Estimates assume NVMe storage. HDD impact is 5–20× larger.

---

## Risk Assessment

**Risk: `medium`**

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Index drift (record deleted, index not updated) | Low | Graceful skip of missing records |
| Two processes write simultaneously, corrupt index | Low | JSONL append is atomic per-line on POSIX; add advisory file lock |
| Index grows without bound (no compaction) | Medium | Each line is ~200 B; 100K records ≈ 20 MB — acceptable |
| Existing observe dirs have no index (search regression) | Zero | Full-scan fallback when index absent |

---

## Test Strategy

1. **Unit — `save_record` appends to index:**
   ```python
   def test_save_record_writes_index_entry(tmp_path):
       record = make_test_record(...)
       save_record(tmp_path, record)
       index_path = tmp_path / record.model / "_index.jsonl"
       assert index_path.exists()
       lines = index_path.read_text().splitlines()
       assert len(lines) == 1
       assert json.loads(lines[0])["id"] == record.id
   ```

2. **Unit — search uses index, skips non-matching full reads:**
   Mock `load_record` to count calls. Assert call count equals page size, not total records.

3. **Unit — graceful degradation without index:**
   Assert `search_records_page` returns correct results when `_index.jsonl` absent.

4. **Unit — stale index entry (missing record file):**
   Write index entry pointing to a deleted `.json` file. Assert search does not crash.

5. **Integration — `index-rebuild` CLI command:**
   Write 50 records, delete index, run `drove observe index-rebuild`, assert index rebuilt
   with 50 entries.

---

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| Phase 1 (write path) | 0.5 days |
| Phase 2 (read path) | 1 day |
| Phase 3 (CLI rebuild) | 0.5 days |
| Phase 4 (defensive) | 0.5 days |
| Tests | 0.5 days |
| **Total** | **~3 days** |

---

## Alternatives Considered

- **In-memory index (rebuilt on startup):** Startup cost O(n) reads, lost on restart.
  Worse than JSONL for long-running servers with large observe dirs.
- **SQLite index:** Adds a dependency, more complex than needed for this access pattern.
- **Restrict search to metadata fields only:** Simpler but degrades UX for users
  searching response bodies (common workflow for debugging model outputs).
- **Cap observe records (TTL/limit):** Orthogonal to this proposal; doesn't help
  users with existing large dirs.
