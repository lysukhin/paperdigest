# Design: Index-Based Digests + Idempotent Pipeline

**Date:** 2026-02-28

## Goals

1. ~~Add `./pd clean` command~~ — Already exists with `--db`, `--digests`, `--pwc`, `--all`, `-y` flags.
2. Replace date-based digest naming (`digest_2026-02-28.md`) with index-based (`digest_001.md`).
3. Ensure expensive operations (LLM filter, LLM summarize) run at most once per paper.

## Part 1: Index-Based Digest Naming

### DB Schema

Add `digest_number INTEGER` column to `digests` table. Auto-assigned as `MAX(digest_number) + 1` in `log_digest()`. The `date` column stays for metadata/display but no longer drives filenames.

### Migration

New migration function `migrate_add_digest_number()`:
- Adds `digest_number` column if missing
- Backfills existing rows with sequential numbers ordered by `created_at`

### Model

Add `number: int` field to `Digest` dataclass. The `date` field stays (used in template headers).

### File Naming

`deliver_markdown()` accepts digest number, generates `digest_NNN.md` (zero-padded to 3 digits).

### CLI Orchestration Change

Currently `log_digest()` happens after markdown delivery. Must reorder: call `log_digest()` first to get the digest number, then pass it to `deliver_markdown()` and `deliver_telegram()`.

New flow in `_cmd_digest_inner`:
```
digest_number = db.log_digest(paper_ids, status="pending")
md_path = deliver_markdown(digest, config)  # digest.number set
deliver_telegram(digest, config)            # uses digest.number
db.update_digest_status(digest_number, "delivered" | "partial_delivery")
```

### Web Routes

- Route changes from `/digest/{date}` to `/digest/{number}`
- Validation changes from `\d{4}-\d{2}-\d{2}` regex to `\d+`
- `_parse_digest_meta()` parses `digest_(\d+)` from filenames
- `_render_digest()` constructs path as `digest_{number:03d}.md`
- List page shows `#1`, `#2`, etc.

### Telegram

Link changes from `/digest/2026-02-28` to `/digest/1`.

## Part 2: Idempotent Pipeline (Skip Already-Processed Papers)

### Filter — Already Idempotent

`filter.py:189-199` already checks `get_latest_filter_result()` and skips LLM calls for papers with cached results in `paper_filter_results` table.

### Enrich — Already Idempotent

`cli.py:166` filters with `p.citations is None`, skipping already-enriched papers.

### Score — Cheap, Always Rerun

Pure math (no API calls), uses upsert. No change needed.

### Summarize — Needs Caching

Currently summaries are computed in-memory, never persisted. On re-run, every paper gets re-summarized (expensive LLM + PDF download).

**New table:**
```sql
CREATE TABLE IF NOT EXISTS paper_summaries (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id),
    one_liner TEXT NOT NULL DEFAULT '',
    affiliations TEXT NOT NULL DEFAULT '',
    method TEXT NOT NULL DEFAULT '',
    data_benchmarks TEXT NOT NULL DEFAULT '',
    key_results TEXT NOT NULL DEFAULT '',
    novelty TEXT NOT NULL DEFAULT '',
    ad_relevance TEXT NOT NULL DEFAULT '',
    limitations TEXT NOT NULL DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);
```

**New DB methods:**
- `get_summary(paper_id: int) -> Summary | None`
- `upsert_summary(paper_id: int, summary: Summary)`

**Summarizer change:** `summarize_paper()` checks DB first via `db.get_summary(paper.db_id)`. Returns cached summary if found. Persists new summaries after successful LLM call via `db.upsert_summary()`.

## Files to Change

| File | Change |
|------|--------|
| `db.py` | Add `paper_summaries` table to schema, `digest_number` to digests, new methods |
| `migrate.py` | `migrate_add_digest_number()` with backfill |
| `models.py` | Add `number: int` to `Digest` |
| `cli.py` | Reorder delivery to get digest number first; pass to delivery functions |
| `delivery/markdown.py` | Use `digest.number` for filename |
| `delivery/telegram.py` | Use `digest.number` in URL |
| `web.py` | Route `/digest/{number}`, parse `digest_NNN.md` filenames |
| `summarizer.py` | Check DB for cached summaries before LLM calls |
| `test_db.py` | Test new methods and schema |
| `test_summarizer.py` | Test summary caching |
| `test_telegram.py` | Update URL assertions |
| `test_config.py` | If needed |
