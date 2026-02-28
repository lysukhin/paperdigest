# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**paperdigest** — an automated research paper digest system that fetches papers from arXiv, conference proceedings (DBLP), and lab blogs (NVIDIA, Waymo), filters them for relevance using a cheap LLM, enriches survivors with citation/code data (Semantic Scholar, Papers with Code), scores by quality, generates full-text LLM summaries with ranking, and delivers digests as Markdown files, Telegram messages, or a web dashboard. Currently configured for VLM/VLA for Autonomous Driving but works for any research topic via `config.yaml`.

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
python -m pytest tests/test_scoring.py::TestQualityScoring::test_venue_tier -v

# CLI (all commands accept --config <path>, default: config.yaml)
python -m paperdigest init --skip-pwc     # create DB, optionally download PWC links
python -m paperdigest run                  # full pipeline: fetch → digest
python -m paperdigest fetch                # collect papers from arXiv
python -m paperdigest filter               # run LLM relevance filtering
python -m paperdigest enrich               # add Semantic Scholar + PWC data
python -m paperdigest score                # compute quality scores
python -m paperdigest digest --dry-run     # generate digest without delivering (runs filter→enrich→score→summarize→rank)
python -m paperdigest serve                # start web dashboard (localhost:8000)
python -m paperdigest stats                # show DB and LLM usage statistics
python -m paperdigest setup               # interactive setup wizard
python -m paperdigest -v <subcommand>      # verbose/debug logging
./pd run                                   # shorthand wrapper for python -m paperdigest
```

## Architecture

### Pipeline Flow
```
arXiv + Blogs + DBLP → Dedup → SQLite → LLM Filter → Semantic Scholar + PWC → Quality Score → [LLM Summary + Rank] → Markdown / Telegram / Web
       (fetch)         (batch)  (store)   (filter)          (enrich)           (score)          (summarize)            (deliver)
```

### Package Layout (`src/paperdigest/`)

- **cli.py** — argparse-based CLI with 11 subcommands (`run`, `fetch`, `filter`, `enrich`, `score`, `digest`, `init`, `serve`, `stats`, `setup`, `clean`), global `-v` flag; `run` calls `fetch` then `digest`; `digest` orchestrates filter→enrich→score→summarize→rank
- **config.py** — YAML loading into validated dataclasses; loads secrets from `.env` file (falls back to env vars: `LLM_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`, `OPENAI_ADMIN_KEY`, `TELEGRAM_*`); `topic.description` for LLM filter; `LLMConfig` split into `FilterLLMConfig` + `SummarizerLLMConfig`; `EnrichmentConfig` for toggling Semantic Scholar
- **models.py** — core data models: `Paper`, `Scores`, `Summary`, `DigestEntry`, `Digest`, `FilterResult`; `Scores` has `quality` + `llm_rank` (no `relevance`/`final`); `Digest` has `rejected` field
- **db.py** — SQLite with WAL mode; context manager (`with Database(...) as db:`); 5 tables (`papers`, `scores`, `digests`, `llm_usage`, `paper_filter_results`); upsert patterns, cost tracking
- **filter.py** — LLM-based paper relevance filtering using cheap model (gpt-4o-mini); reads title+abstract against `topic.description`; binary relevant/not with reason; fail-open on errors; cost tracked separately with `filter_` prefix on `run_id`
- **dedup.py** — 4-stage deduplication: exact arXiv ID → exact DOI → fuzzy title within batch → fuzzy title against DB (SequenceMatcher, 0.85 threshold)
- **scoring.py** — quality score only (venue tier, h-index, citations, code, freshness); relevance scoring removed (handled by LLM filter)
- **summarizer.py** — OpenAI-compatible LLM with structured JSON output; always uses full-text PDF (abstract fallback); adds `rank_papers()` method for LLM-based ranking of survivors; uses `config.llm.summarizer`; per-run and monthly cost caps with graceful degradation
- **pdf.py** — PDF download and text extraction via PyMuPDF for full-text summarization
- **collectors/** — abstract `BaseCollector` interface:
  - `arxiv.py` — arXiv API with query building, rate limiting (3s delay, 3 retries), per-query result count logging
  - `nvidia.py` — NVIDIA Developer Blog via RSS feed, filtered by AV keywords
  - `waymo.py` — Waymo Research page scraper, extracts arXiv IDs from links
  - `wayve.py` — Wayve Science scraper (currently blocked by WAF, disabled in config)
  - `dblp.py` — DBLP conference proceedings search; covers CVPR, ICRA, IROS, ECCV, IV, ITSC, etc. (NeurIPS excluded — triggers DBLP API 500)
- **web.py** — FastAPI read-only web dashboard with digest archive and digest viewer; renders markdown digests to HTML (uses `md_in_html` extension for `<details>` blocks); `WebConfig.public_url` used by Telegram for digest links
- **usage.py** — fetches real OpenAI token usage and USD costs via Admin API (`/v1/organization/usage/completions` + `/v1/organization/costs`); requires `OPENAI_ADMIN_KEY`
- **enrichment/** — `semantic_scholar.py` (citations, h-index, venue, OA PDF) and `pwc.py` (code links via local JSON dump)
- **delivery/** — `markdown.py` (sandboxed Jinja2 template at `templates/digest.md.j2`) and `telegram.py` (raw HTTP, MarkdownV2, compact top-5 format with inline keyboard button linking to web digest)
- **setup.py** — interactive setup wizard: offers `config.yaml.example` as default or custom topic; generates config.yaml, .env, Caddyfile, crontab; detects IP vs FQDN for Caddyfile (no TLS for IPs) and `public_url`; tests API connections; initializes DB

### Key Design Decisions

- **LLM filter replaces keyword scoring** — `topic.description` (natural language) tells the LLM what's relevant; keywords stay for arXiv query building only
- **Two-tier LLM architecture** — cheap model (gpt-4o-mini) for binary filter, good model (gpt-5-nano) for full-text summary + ranking
- **Fail-open filter** — budget exhaustion or LLM errors treat papers as relevant
- **Always full-text summarization** — PDF fetched for all survivors, abstract fallback
- **Rejected papers tracked** — stored in `paper_filter_results` table, shown in digest footer
- **Collector extensibility** — `BaseCollector` ABC allows adding new paper sources beyond arXiv
- **Local PWC lookup** — downloads full JSON dump once (`init`), then does instant local lookups instead of per-paper API calls
- **Scoring is configurable** — quality weights, venue tiers in `config.yaml`
- **Individual API failures don't break the pipeline** — enrichment and summarization handle errors per-paper gracefully
- **Enrichment toggle** — Semantic Scholar can be disabled via config (fresh papers have 0 citations); PWC enrichment is always on (local lookup)
- **Configurable prompt instructions** — extra_instructions field on filter and summarizer configs appended to system prompts; allows steering LLM output without replacing base prompts
- **Telegram as notification channel** — compact top-5 digest with inline keyboard button linking to full web digest; falls back to text link if Telegram rejects the button URL (e.g. non-public URLs); `web.public_url` controls the link target

### Database

SQLite at `data/papers.db`. Key tables:
- `papers` — canonical record per paper, unique on `arxiv_id`, indexed on `doi`
- `scores` — one row per paper, `quality REAL` + `llm_rank INTEGER` (1 = best, 0 = unranked)
- `paper_filter_results` — per-paper filter decisions: `paper_id`, `run_date`, `relevant`, `reason`
- `llm_usage` — per-run cost tracking with `run_id` unique constraint

### Test Structure

Unit tests across 10 files — all with no external API calls:
- `test_config.py` — config loading, validation, defaults, error cases, weight validation, `public_url` parsing
- `test_db.py` — database operations
- `test_dedup.py` — title normalization, fuzzy matching, ID/DOI/title dedup (batch + DB)
- `test_filter.py` — LLM filter with mocked OpenAI client, fail-open behavior, budget enforcement, cost tracking
- `test_models.py` — data model tests
- `test_progress.py` — progress display tests
- `test_scoring.py` — quality scoring with `pytest.approx`, freshness decay, ranking
- `test_setup.py` — setup wizard generation functions, IP address detection, Caddyfile generation
- `test_summarizer.py` — LLM summarization with mocked OpenAI client, budget enforcement, cost tracking, ranking, error handling
- `test_telegram.py` — Telegram message formatting (compact top-5), inline button presence/absence, text link fallback, error handling

### Docker

- `Dockerfile` — multi-stage build (builder + runtime), includes supercronic for container cron; copies `config.yaml.example` into image
- `docker-compose.yml` — three services: `web` (uvicorn), `cron` (supercronic), `caddy` (reverse proxy on ports 38080/38443); web service mounts are writable for setup wizard
- `config.yaml.example` — full Autonomous Driving config (the default); copied by setup wizard to `config.yaml`
- `config.yaml` — gitignored, user's local copy (generated by setup or `cp config.yaml.example config.yaml`)
- `Caddyfile` — generated by setup wizard; uses `:80` for IPs (no TLS), FQDN for domains (auto-HTTPS)
- `crontab` — generated by setup wizard, used by supercronic in cron container

### Docs

`docs/` contains detailed references (linked from README):
- `configuration.md` — full config.yaml reference with all YAML blocks
- `scoring.md` — scoring formulas, LLM filter/ranking details, enrichment sources, cost controls
- `roadmap.md` — planned features and TODO items
- `plans/` — design and implementation plans for features

## Runtime Data

`data/` directory (gitignored) contains:
- `papers.db` — SQLite database
- `digests/` — generated markdown digests (`digest_YYYY-MM-DD.md`)
- `pwc_links.json` — Papers with Code lookup cache

## Secrets

Copy `.env.example` to `.env` and fill in values. Copy `config.yaml.example` to `config.yaml` and edit for your topic. The `.env` file is loaded automatically from the same directory as `config.yaml`. Environment variables take precedence over `.env` values.

## Code Conventions

- `Database` is a context manager — always use `with Database(...) as db:`
- Config validation enforces quality weights summing to 1.0 — tests that override partial weights must provide all 5 weights
- Telegram uses MarkdownV2 — escape user-controlled strings with `_escape_markdown()`
- Scoring uses `paper.published.astimezone(timezone.utc)` for timezone-safe freshness
- Tests use `tmp_path` fixture for temp files/DBs, never `tempfile.mktemp`
- Tests use `pytest.approx()` for deterministic float assertions
- Non-arXiv papers use synthetic IDs: `dblp:conf/cvpr/key`, `nvidia:slug`, `waymo:slug` — dedup handles cross-source overlap via fuzzy title matching
- Blog/DBLP collectors handle API failures gracefully (return empty list, pipeline continues)
- LLM filter uses `config.llm.filter.*`, summarizer uses `config.llm.summarizer.*`
- LLM config supports `null` temperature and `max_completion_tokens` — omitted from API call when null (required for reasoning models like gpt-5-nano)
- Filter has `filter_` prefix on `run_id` for cost tracking separation
- `Scores.llm_rank` — 1 = best, 0 = unranked
- Telegram inline button requires public HTTPS URL; `http://127.0.0.1` works for local dev, `localhost` does not (Telegram client limitation)
