# paperdigest

Automated research paper digest system. Fetches fresh academic papers from arXiv, enriches them with citation and code data, scores them by relevance and quality, optionally generates LLM summaries, and delivers a ranked digest — all from a single CLI command.

Ships configured for **VLM/VLA for Autonomous Driving** but works for any research topic via `config.yaml`.

## How It Works

```
arXiv API ──→ Dedup ──→ SQLite ──→ Semantic Scholar + PWC ──→ Score ──→ [LLM Summary] ──→ Markdown / Telegram
  (fetch)               (store)         (enrich)             (score)     (optional)        (deliver)
```

1. **Collect** — Query arXiv by keywords and categories, filtered to a configurable lookback window
2. **Dedup** — Remove duplicates by arXiv ID, DOI, or fuzzy title matching (85% similarity threshold)
3. **Store** — Upsert papers into a local SQLite database
4. **Enrich** — Pull citation counts, author h-indices, venue info, and open access PDFs from Semantic Scholar; check Papers with Code for code repositories
5. **Score** — Rank papers using a weighted combination of relevance (keyword matching) and quality (venue tier, citations, author reputation, code availability, freshness)
6. **Summarize** *(optional)* — Send top-N papers to an OpenAI-compatible LLM for structured summaries with built-in cost controls
7. **Deliver** — Write a Markdown digest file; optionally push to Telegram

The pipeline works end-to-end **without LLM** — the default mode produces ranked papers with metadata but no prose summaries.

## Quick Start

```bash
# Install
pip install -e .

# Initialize database (skip PWC download for now)
python -m paperdigest init --skip-pwc

# Run full pipeline
python -m paperdigest run --config config.yaml

# Or run stages individually
python -m paperdigest fetch --config config.yaml
python -m paperdigest enrich --config config.yaml
python -m paperdigest score --config config.yaml
python -m paperdigest digest --config config.yaml --dry-run
```

Output lands in `data/digests/digest_YYYY-MM-DD.md`.

### Cron Setup

```bash
0 8 * * * cd /path/to/whatscooking && python -m paperdigest run --config config.yaml
```

## Installation

Requires **Python 3.9+**.

```bash
# Core (no LLM)
pip install -e .

# With LLM summarization
pip install -e ".[llm]"

# With dev/test dependencies
pip install -e ".[dev]"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `arxiv` | arXiv API client |
| `requests` | HTTP for Semantic Scholar, PWC, Telegram |
| `pyyaml` | Config file parsing |
| `jinja2` | Digest template rendering |
| `openai` *(optional)* | LLM summarization |

## CLI Reference

All commands accept `--config <path>` (default: `config.yaml`). The global `-v`/`--verbose` flag enables debug logging and must appear **before** the subcommand.

```
python -m paperdigest [-v] <command> [--config config.yaml] [options]
```

| Command | Description | Extra Flags |
|---------|-------------|-------------|
| `run` | Full pipeline: fetch → enrich → score → digest | |
| `fetch` | Collect papers from arXiv | |
| `enrich` | Enrich stored papers with Semantic Scholar + PWC data | |
| `score` | Score all papers in the database | |
| `digest` | Generate and deliver digest | `--dry-run` |
| `init` | Create database and download PWC links | `--skip-pwc` |
| `stats` | Show database and LLM usage statistics | |

### Examples

```bash
# Verbose fetch to debug arXiv queries
python -m paperdigest -v fetch --config config.yaml

# Generate digest without LLM calls ($0 cost)
python -m paperdigest digest --config config.yaml --dry-run

# Check how many papers are in the DB and monthly LLM spend
python -m paperdigest stats --config config.yaml

# Init DB without downloading the ~500MB PWC dump
python -m paperdigest init --skip-pwc --config config.yaml
```

## Configuration

Everything is controlled by `config.yaml`. Here's the full structure with defaults:

### Topic

```yaml
topic:
  name: "VLM/VLA for Autonomous Driving"
  primary_keywords:                # At least one required
    - "vision language model autonomous driving"
    - "VLM autonomous driving"
  secondary_keywords:              # Optional, boost relevance score
    - "end-to-end driving"
    - "scene understanding driving"
  benchmarks:                      # Optional, boost relevance score
    - "nuScenes"
    - "CARLA"
  arxiv_categories:                # Filters arXiv queries
    - "cs.CV"
    - "cs.AI"
    - "cs.RO"
    - "cs.LG"
```

### Collection

```yaml
collection:
  lookback_days: 7     # How far back to search
  max_results: 200     # Max papers per arXiv query
```

### Scoring

```yaml
scoring:
  alpha: 0.65          # final = alpha * relevance + (1 - alpha) * quality

  relevance:
    primary_base: 0.5          # Score for any primary keyword hit
    secondary_increment: 0.1   # Per secondary keyword match
    secondary_cap: 0.3         # Max from secondary keywords
    benchmark_increment: 0.1   # Per benchmark mention
    benchmark_cap: 0.2         # Max from benchmarks

  quality:
    w_venue: 0.25      # Weight for venue tier
    w_author: 0.20     # Weight for author h-index
    w_cite: 0.20       # Weight for citation count
    w_code: 0.15       # Weight for code availability
    w_fresh: 0.20      # Weight for recency

  venue_tiers:
    tier1: [NeurIPS, ICML, ICLR, CVPR, ECCV, ICCV, AAAI, IJCAI, RSS, CoRL]    # → 1.0
    tier2: [IROS, ICRA, WACV, BMVC, ACCV, AISTATS, UAI]                         # → 0.7
    tier3: [IV, ITSC, T-ITS, RA-L, T-RO]                                        # → 0.4
    # Everything else → 0.2
```

### LLM Summarization

Disabled by default. Enable with `llm.enabled: true` and set the `LLM_API_KEY` environment variable.

```yaml
llm:
  enabled: false
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"    # Works with OpenAI, Ollama, OpenRouter, etc.
  cost_control:
    max_cost_per_run: 0.50       # USD — hard stop per run
    max_cost_per_month: 10.00    # USD — refuse if monthly budget exhausted
    warn_at_percent: 80          # Log warning at this % of monthly budget
    input_cost_per_1k: 0.00015   # Update for your model's pricing
    output_cost_per_1k: 0.0006
```

### Delivery

```yaml
delivery:
  markdown:
    enabled: true                # Always-on local file archive
  telegram:
    enabled: false               # Set true + env vars to enable

digest:
  top_n: 20                      # Papers in the digest
  summarize_top_n: 15            # Papers to send to LLM (when enabled)
  output_dir: "data/digests"
```

### Database & PWC

```yaml
database:
  path: "data/papers.db"

pwc:
  links_path: "data/pwc_links.json"
```

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `LLM_API_KEY` | If `llm.enabled: true` | OpenAI or compatible API key |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Faster rate limits on Semantic Scholar (0.5s vs 3.5s delay) |
| `TELEGRAM_BOT_TOKEN` | If Telegram enabled | From [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | If Telegram enabled | Target channel or group ID |

## Scoring System

### Relevance Score (0–1)

Measures how well a paper matches your configured topic keywords.

```
score = 0

If any primary keyword appears in title+abstract:
    score += 0.5

For each secondary keyword match:
    score += 0.1  (capped at 0.3 total)

For each benchmark mention:
    score += 0.1  (capped at 0.2 total)

Maximum: 1.0
```

Keyword matching is case-insensitive substring search against the concatenation of title and abstract.

### Quality Score (0–1)

Measures paper quality from external signals, independent of topic.

```
quality = w_venue  * venue_tier(venue)              # 1.0 / 0.7 / 0.4 / 0.2
        + w_author * min(1, max_hindex / 50)        # Author reputation
        + w_cite   * min(1, log(1 + citations) / 5) # Citation impact
        + w_code   * (1.0 if has_code else 0.0)     # Reproducibility signal
        + w_fresh  * max(0, 1 - age_days / 30)      # Linear decay over 30 days
```

### Final Score

```
final = alpha * relevance + (1 - alpha) * quality
```

Default `alpha = 0.65` — relevance-weighted, but quality still matters.

## LLM Summarization

When enabled, the top-N papers get structured summaries with these fields:

| Field | Description |
|-------|-------------|
| `one_liner` | Core contribution in one sentence (max 150 chars) |
| `method` | Proposed method/approach (2–3 sentences) |
| `data_benchmarks` | Datasets and benchmarks used |
| `key_results` | Most important results |
| `novelty` | What's new vs. prior work |
| `ad_relevance` | Relevance to autonomous driving |
| `limitations` | Key limitations or open questions |

### Cost Controls

The LLM is the only component that costs money. Multiple safeguards are built in:

- **Per-run limit** (`max_cost_per_run: $0.50`) — stops summarizing if exceeded mid-run
- **Monthly limit** (`max_cost_per_month: $10.00`) — refuses to call LLM if monthly budget exhausted
- **Warning threshold** (`warn_at_percent: 80`) — logs a warning at 80% of monthly spend
- **Dry-run mode** (`--dry-run`) — runs scoring but skips all LLM calls
- **Graceful degradation** — if any LLM call fails, the paper still appears in the digest without a summary

All token usage is tracked in the `llm_usage` SQLite table and visible via `stats`.

**Typical costs** (gpt-4o-mini, ~15 papers/run):
- ~$0.003 per paper → ~$0.05 per run → ~$1.50/month at daily runs
- Default $10/month limit gives ~6x headroom

### Compatible Providers

The `base_url` config works with any OpenAI-compatible API:

```yaml
# OpenAI (default)
base_url: "https://api.openai.com/v1"

# Ollama (local, free)
base_url: "http://localhost:11434/v1"

# OpenRouter
base_url: "https://openrouter.ai/api/v1"
```

## Enrichment Sources

### Semantic Scholar

Queries the [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/) for each paper:

- **Citation count** — used in quality scoring
- **Author h-indices** — max h-index across all authors
- **Venue** — conference/journal name for tier classification
- **Open access PDF** — direct link when available

Rate limits: 0.5s between requests with an API key, 3.5s without. Automatically retries on 429 responses with a 60s backoff.

### Papers with Code

Uses the [PWC links dump](https://paperswithcode.com/about) (~500MB gzipped JSON) to look up:

- **Code repository URL** — GitHub link
- **Official flag** — whether the repo is from the paper's authors

Run `python -m paperdigest init` to download the dump. Afterwards, lookups are instant (local JSON file).

## Deduplication

Three-stage dedup runs on every `fetch`:

1. **arXiv ID** — exact match against database and current batch
2. **DOI** — exact match against database (catches re-submissions)
3. **Fuzzy title** — `difflib.SequenceMatcher` with 0.85 threshold within the current batch (catches minor title variations like "Model" vs "Models")

## Database

SQLite with WAL mode. Four tables:

| Table | Purpose |
|-------|---------|
| `papers` | Canonical record per paper — arXiv ID (unique), title, abstract, authors, enrichment fields |
| `scores` | Latest relevance/quality/final scores per paper |
| `digests` | Log of generated digests with paper IDs and delivery status |
| `llm_usage` | Cost tracking per run — date, tokens, estimated USD |

Located at `data/papers.db` by default (gitignored).

## Project Structure

```
whatscooking/
├── config.yaml                      # Topic, scoring, LLM, delivery settings
├── pyproject.toml                   # Package metadata + dependencies
├── src/paperdigest/
│   ├── __init__.py
│   ├── __main__.py                  # python -m paperdigest entry
│   ├── cli.py                       # argparse CLI with 7 commands
│   ├── config.py                    # YAML → validated dataclasses
│   ├── models.py                    # Paper, Scores, Summary, Digest dataclasses
│   ├── db.py                        # SQLite schema, CRUD, cost tracking
│   ├── dedup.py                     # arXiv ID / DOI / fuzzy title dedup
│   ├── scoring.py                   # Relevance + Quality → Final score
│   ├── summarizer.py                # LLM summaries with cost controls
│   ├── collectors/
│   │   ├── base.py                  # Abstract collector interface
│   │   └── arxiv.py                 # arXiv API collection
│   ├── enrichment/
│   │   ├── semantic_scholar.py      # Citations, h-index, venue, OA PDF
│   │   └── pwc.py                   # Papers with Code → code links
│   ├── delivery/
│   │   ├── markdown.py              # Local .md file output
│   │   └── telegram.py              # Telegram Bot API delivery
│   └── templates/
│       └── digest.md.j2             # Jinja2 digest template
├── tests/
│   ├── test_scoring.py              # 10 tests — relevance, quality, ranking
│   ├── test_dedup.py                # 9 tests — normalization, matching, dedup
│   └── test_config.py               # 8 tests — valid/invalid YAML, defaults
└── data/                            # Runtime, gitignored
    ├── papers.db
    ├── pwc_links.json
    └── digests/
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

29 tests covering:
- **Scoring** — keyword matching, quality signals, ranking order, score capping
- **Dedup** — exact ID/DOI, fuzzy title matching, batch dedup, DB dedup
- **Config** — valid loading, missing required fields, custom overrides, defaults

## Adapting to a Different Topic

Edit `config.yaml`:

1. Change `topic.name`, `primary_keywords`, `secondary_keywords`, and `benchmarks` to your field
2. Update `arxiv_categories` to match (see [arXiv taxonomy](https://arxiv.org/category_taxonomy))
3. Adjust `venue_tiers` for your field's top conferences
4. Tune `scoring.alpha` — higher values (→ 1.0) weight keyword relevance more; lower values weight quality signals more
5. Run `python -m paperdigest init --skip-pwc && python -m paperdigest run`

## TODO — Roadmap to Production

Features described in `deep-research-report.md` that are not yet implemented, organized by priority tier.

### Collection — Additional Sources

- [ ] **Crossref REST API collector** — DOI-level metadata, publication status (preprint → published), funding/license/ORCID/ROR fields. Free API with polite-usage rate limits (changed Dec 2025). Gives the strongest "is this published?" signal for journal papers.
- [ ] **OpenAlex collector** — Unified scholarly graph (CC0 metadata). Works/authors/institutions/sources in one API. Credits/key system; Premium tier unlocks `from_updated_date` filter. Abstracts come as inverted index (needs reconstruction to plain text).
- [ ] **OpenReview collector** — Best source for acceptance decisions at conferences hosted on OpenReview (ICLR, NeurIPS workshops, etc.). Official `openreview-py` client exists. Beware v1/v2 API differences.
- [ ] **CVF Open Access scraper** — Accepted-version PDFs for CVPR/ICCV/ECCV. Strong acceptance signal. No official API; requires HTML parsing.
- [ ] **IEEE Xplore API** — Final proceedings for IEEE conferences/journals. Requires API key application and approval. Enterprise-oriented.
- [ ] **ACL Anthology** — NLP venue metadata from public GitHub repo + Python module. Useful if topic overlaps with NLP (language-guided navigation, etc.).
- [ ] **arXiv OAI-PMH / RSS** — Alternative to the search API for bulk daily collection. RSS updates daily per category; OAI-PMH gives incremental harvest.

### Enrichment — Additional Signals

- [ ] **GitHub API enrichment** — Stars, commit frequency, issues, releases for code repos. Adds depth beyond PWC's binary "has code" signal. Requires token; rate limits apply per endpoint.
- [ ] **Unpaywall enrichment** — Canonical OA status + best OA PDF link. REST API (~100k req/day free). Data Feed (daily changefiles) available by subscription.
- [ ] **Influential citations** — Semantic Scholar exposes `influentialCitationCount` (ML-classified by citation context). More meaningful than raw citation count for fresh papers.
- [ ] **OpenAlex citation cross-check** — `cited_by_count` as a second opinion on citation data, especially for DOI-registered works.
- [ ] **Author affiliation scoring** — Crossref ROR/affiliations + OpenAlex institutions. Build an `AuthorCredibilityScore` with bonus for top-tier affiliations (configurable list).
- [ ] **OpenCitations** — Open citation graph as an independent citation data source. Useful for bibliometric analysis layer.

### Scoring — Advanced Relevance

- [ ] **Embedding-based relevance** — Compute title+abstract embeddings and measure cosine similarity against a seed set of known-relevant papers. More robust than keyword matching for catching novel terminology.
- [ ] **LLM-based topic classification** — "Is this paper about VLM/VLA applied to autonomous driving?" as a supplementary relevance signal with validation against keyword scores.
- [ ] **Publication status score** — Factor acceptance/publication status into quality: accepted > preprint. Sources: CVF pages, OpenReview decisions, Crossref venue field.
- [ ] **Influential citation velocity** — Track citation growth at 7/30/90 days. More useful than absolute count for fresh papers where citations ≈ 0.
- [ ] **Configurable venue prior** — Per-venue relevance prior for conferences that reliably publish AD/VLM/VLA work (e.g., CVPR perception track, CoRL).

### Scoring — Incremental Re-evaluation

- [ ] **Re-enrichment scheduler** — Periodically re-fetch citations/venue/code for existing papers (e.g., weekly for papers < 30 days old, monthly for < 90 days). Requires tracking `last_enriched_at` and a re-enrichment CLI command/cron job.
- [ ] **Status transition tracking** — Detect and log when a paper moves from preprint → accepted → published. Store events in a `paper_events` table for audit.

### Summarization — Full-Text & Validation

- [ ] **Full-text PDF summarization** — Download and parse PDFs for top-N papers. Use Grobid or PyMuPDF for extraction. Much richer summaries (method details, tables, experimental setup) but adds cost (OpenAlex charges 100 credits/file for TEI XML) and storage requirements.
- [ ] **Summary template versioning** — Store `SummaryTemplate` as a versioned object in the DB. Each generated summary is linked to a template version for auditability. Templates as Jinja2 or YAML, editable without code changes.
- [ ] **Schema validation on LLM output** — Strict JSON schema check (field types, max lengths, required fields) on every LLM response. Reject and retry on malformed output.
- [ ] **Anti-hallucination cross-checks** — If LLM claims "accepted at CVPR", verify against CVF/OpenReview/Crossref data. If LLM claims "code available", verify against PWC/GitHub. Flag unverified claims in the digest.
- [ ] **Provenance tracking** — Record which API source provided each enrichment field (e.g., "citationCount from Semantic Scholar 2026-02-20"). Stored per-field, queryable for debugging and trust assessment.

### Delivery — Additional Channels & Formats

- [ ] **Email delivery (SMTP)** — HTML email with clickable table of contents, top-N highlighted, full list below. Most universal channel for teams.
- [ ] **Slack delivery** — Post digest to a Slack channel via `chat.postMessage` Web API. Rich formatting with blocks/attachments.
- [ ] **HTML report generation** — Rendered HTML digest with styling, anchor links, expandable summaries. Suitable for email body or web archive.
- [ ] **PDF report generation** — Archival-quality PDF from HTML template. Useful for sharing outside messaging platforms.
- [ ] **Web dashboard / archive** — Browse historical digests, search papers, filter by score/date/venue. Simple Flask/FastAPI app or static site generator.

### Infrastructure — Production Hardening

- [ ] **PostgreSQL migration** — Replace SQLite with Postgres for concurrent access, better full-text search, and production reliability.
- [ ] **Vector index** — pgvector or dedicated vector store for embedding-based search and relevance scoring.
- [ ] **Object storage** — S3-compatible storage for downloaded PDFs/TEI XML. Respect arXiv's policy: store extracted text/embeddings, link to arXiv for PDFs.
- [ ] **Task queue** — Redis + Celery/RQ to parallelize network I/O (API calls) and LLM calls across workers.
- [ ] **Orchestration** — Airflow or Prefect DAGs instead of flat cron. Enables retries, dependency graphs, backfill, and monitoring.
- [ ] **Centralized logging & metrics** — Structured logs, API call latency/error tracking, scoring distribution histograms, alerting on failures.
- [ ] **Secrets management** — Vault/KMS for API keys instead of environment variables.
- [ ] **Audit trail** — Log every delivery (who received what, when) for compliance and debugging.

### Enterprise

- [ ] **RBAC / SSO / SCIM** — Role-based access, corporate SSO integration.
- [ ] **Data isolation & compliance** — Tenant isolation, retention policies, GDPR considerations.
- [ ] **On-prem LLM endpoint** — Private LLM deployment for sensitive/proprietary research contexts.
- [ ] **Subscription data feeds** — Unpaywall Data Feed (daily changefiles), IEEE API key for full proceedings access.

### Quality of Life

- [ ] **Zotero integration** — Export digest entries to a Zotero collection via Web API. Useful for researchers who manage reading lists in Zotero.
- [ ] **Hugging Face Hub integration** — Check for associated models/datasets on HF Hub. PWC dumps are also available as HF datasets.
- [ ] **Google Scholar alerts (manual)** — Document a human-in-the-loop workflow for GS alerts as a supplementary signal. No API exists; scraping is unreliable (CAPTCHA/429) and risks ToS violation.
- [ ] **Configurable digest frequency** — Support weekly/biweekly digests in addition to daily, with appropriate dedup windows.
- [ ] **Interactive score tuning** — CLI or notebook for adjusting weights and previewing how rankings change on existing data.

## License

MIT
