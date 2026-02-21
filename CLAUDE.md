# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**paperdigest** — an automated research paper digest system that fetches papers from arXiv, conference proceedings (DBLP), and lab blogs (NVIDIA, Waymo), enriches them with citation/code data (Semantic Scholar, Papers with Code), scores by relevance and quality, optionally generates LLM summaries (with full-text PDF support), and delivers digests as Markdown files, Telegram messages, or a web dashboard. Currently configured for VLM/VLA for Autonomous Driving but works for any research topic via `config.yaml`.

## Commands

```bash
# Install
pip install -e .              # core only
pip install -e ".[llm]"       # with LLM summarization
pip install -e ".[web]"       # with web dashboard
pip install -e ".[dev]"       # with dev/test tools

# Run tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_scoring.py -v

# Run a single test
python -m pytest tests/test_scoring.py::TestRelevanceScoring::test_primary_keyword_hit -v

# CLI (all commands accept --config <path>, default: config.yaml)
python -m paperdigest init --skip-pwc     # create DB, optionally download PWC links
python -m paperdigest run                  # full pipeline: fetch → enrich → score → digest
python -m paperdigest fetch                # collect papers from arXiv
python -m paperdigest enrich               # add Semantic Scholar + PWC data
python -m paperdigest score                # compute scores
python -m paperdigest digest --dry-run     # generate digest without delivering
python -m paperdigest serve                # start web dashboard (localhost:8000)
python -m paperdigest stats                # show DB and LLM usage statistics
python -m paperdigest -v <subcommand>      # verbose/debug logging
./wc run                                   # shorthand wrapper for python -m paperdigest
```

## Architecture

### Pipeline Flow
```
arXiv + Blogs + DBLP → Dedup → SQLite → Semantic Scholar + PWC → Score → [LLM Summary] → Markdown / Telegram / Web
       (fetch)         (batch)  (store)       (enrich)           (score)   (optional)      (deliver)
```

### Package Layout (`src/paperdigest/`)

- **cli.py** — argparse-based CLI with 8 subcommands (`run`, `fetch`, `enrich`, `score`, `digest`, `init`, `serve`, `stats`), global `-v` flag
- **config.py** — YAML loading into validated dataclasses; loads secrets from `.env` file (falls back to env vars: `LLM_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `TELEGRAM_*`)
- **models.py** — core data models: `Paper`, `Scores`, `Summary`, `DigestEntry`, `Digest`
- **db.py** — SQLite with WAL mode; context manager (`with Database(...) as db:`); 4 tables (`papers`, `scores`, `digests`, `llm_usage`); upsert patterns, cost tracking
- **dedup.py** — 4-stage deduplication: exact arXiv ID → exact DOI → fuzzy title within batch → fuzzy title against DB (SequenceMatcher, 0.85 threshold)
- **scoring.py** — relevance score (keyword matching in title+abstract) + quality score (venue tier, h-index, citations, code, freshness) → weighted final score
- **summarizer.py** — OpenAI-compatible LLM with structured JSON output; supports abstract-only or full-text PDF summarization (`use_full_text` config); per-run ($0.50) and monthly ($10) cost caps with graceful degradation
- **pdf.py** — PDF download and text extraction via PyMuPDF for full-text summarization
- **collectors/** — abstract `BaseCollector` interface:
  - `arxiv.py` — arXiv API with query building, rate limiting (3s delay, 3 retries)
  - `nvidia.py` — NVIDIA Developer Blog via RSS feed, filtered by AV keywords
  - `waymo.py` — Waymo Research page scraper, extracts arXiv IDs from links
  - `wayve.py` — Wayve Science scraper (currently blocked by WAF, disabled in config)
  - `dblp.py` — DBLP conference proceedings search; covers CVPR, ICRA, IROS, ECCV, IV, ITSC, etc. (NeurIPS excluded — triggers DBLP API 500)
- **web.py** — FastAPI web dashboard with digest viewer and pipeline trigger
- **enrichment/** — `semantic_scholar.py` (citations, h-index, venue, OA PDF) and `pwc.py` (code links via local JSON dump)
- **delivery/** — `markdown.py` (sandboxed Jinja2 template at `templates/digest.md.j2`) and `telegram.py` (raw HTTP, MarkdownV2, 4096 char limit)

### Key Design Decisions

- **LLM is optional and off by default** — the system works at zero cost without it; papers still appear in digests without summaries
- **Collector extensibility** — `BaseCollector` ABC allows adding new paper sources beyond arXiv
- **Local PWC lookup** — downloads full JSON dump once (`init`), then does instant local lookups instead of per-paper API calls
- **Scoring is configurable** — all weights, keyword lists, venue tiers, and the relevance/quality balance (`alpha`) are in `config.yaml`
- **Individual API failures don't break the pipeline** — enrichment and summarization handle errors per-paper gracefully

### Database

SQLite at `data/papers.db`. Key tables:
- `papers` — canonical record per paper, unique on `arxiv_id`, indexed on `doi`
- `scores` — one row per paper, `final` score indexed DESC for efficient top-N queries
- `llm_usage` — per-run cost tracking with `run_id` unique constraint

### Test Structure

53 tests across 4 files — all unit tests with no external API calls:
- `test_config.py` — config loading, validation, defaults, error cases, weight/alpha validation
- `test_dedup.py` — title normalization, fuzzy matching, ID/DOI/title dedup (batch + DB)
- `test_scoring.py` — relevance/quality scoring with `pytest.approx`, freshness decay, ranking
- `test_summarizer.py` — LLM summarization with mocked OpenAI client, budget enforcement, cost tracking, error handling

## Runtime Data

`data/` directory (gitignored) contains:
- `papers.db` — SQLite database
- `digests/` — generated markdown digests (`digest_YYYY-MM-DD.md`)
- `pwc_links.json` — Papers with Code lookup cache

## Secrets

Copy `.env.example` to `.env` and fill in values. The `.env` file is loaded automatically from the same directory as `config.yaml`. Environment variables take precedence over `.env` values.

## Code Conventions

- `Database` is a context manager — always use `with Database(...) as db:`
- Config validation enforces `alpha` in [0, 1] and quality weights summing to 1.0 — tests that override partial weights must provide all 5 weights
- Telegram uses MarkdownV2 — escape user-controlled strings with `_escape_markdown()`
- Scoring uses `paper.published.astimezone(timezone.utc)` for timezone-safe freshness
- Tests use `tmp_path` fixture for temp files/DBs, never `tempfile.mktemp`
- Tests use `pytest.approx()` for deterministic float assertions
- Non-arXiv papers use synthetic IDs: `dblp:conf/cvpr/key`, `nvidia:slug`, `waymo:slug` — dedup handles cross-source overlap via fuzzy title matching
- Blog/DBLP collectors handle API failures gracefully (return empty list, pipeline continues)
- LLM config supports `null` temperature and `max_completion_tokens` — omitted from API call when null (required for reasoning models like gpt-5-nano)
- Summarizer has two prompt variants: `SYSTEM_PROMPT_ABSTRACT` and `SYSTEM_PROMPT_FULL_TEXT`, selected by `config.llm.use_full_text`
