# LLM-Driven Filtering and Ranking

**Date:** 2026-02-22
**Status:** Approved

## Problem

The current keyword-based relevance scoring misses papers that use different terminology for the same concepts. A paper about "multimodal foundation models for vehicle navigation" is highly relevant but scores poorly because it doesn't contain the exact keywords "autonomous driving" or "VLM". The system needs semantic understanding of relevance, not string matching.

## Design

### New Pipeline Flow

```
fetch → dedup → store → LLM filter (cheap model) → enrich (survivors) → quality score → PDF fetch → LLM summarize+rank (good model, full text) → digest
```

Compared to the current pipeline:
- **New step:** LLM filter between store and enrich
- **Moved:** Enrichment now runs only on relevant papers (saves API calls)
- **Removed:** Keyword-based relevance scoring
- **Changed:** Summarizer always uses full text (PDF with abstract fallback), and also ranks papers
- **Added:** Rejected papers tracked in DB + shown in digest footer

### Two-Tier LLM Architecture

**Tier 1 — Filter (cheap/fast model, e.g. gpt-4o-mini):**
- Reads each paper's title + abstract
- Compares against a natural language topic description from config
- Outputs: `{relevant: bool, reason: "one sentence"}`
- Fail-open on budget exhaustion (unfiltered papers treated as relevant)
- Tracks cost separately with `filter_` run ID prefix

**Tier 2 — Summarizer + Ranker (good model, e.g. gpt-5-nano):**
- Summarizes each surviving paper from full PDF text (abstract fallback)
- Same 8-field JSON output as today: one_liner, affiliations, method, data_benchmarks, key_results, novelty, ad_relevance, limitations
- After all summaries are done, a final ranking call sends titles + one-liners + quality metadata and asks for an ordered list
- LLM rank is primary sort key, quality score is tiebreaker

### Config Changes

```yaml
topic:
  name: "VLM/VLA for Autonomous Driving"
  description: >
    Papers about vision-language models (VLMs), vision-language-action models (VLAs),
    and multimodal foundation models applied to autonomous driving, including end-to-end
    driving, motion planning with language reasoning, and driving scene understanding.
    Also interested in relevant benchmarks and datasets for evaluating these systems.
  # keywords stay for arXiv query building, NOT for scoring
  primary_keywords: [...]
  secondary_keywords: [...]
  benchmarks: [...]
  arxiv_categories: [...]

llm:
  filter:
    enabled: true
    model: "gpt-4o-mini"
    base_url: "https://api.openai.com/v1"
    temperature: null
    max_completion_tokens: 256
    cost_control:
      max_cost_per_run: 0.10
      max_cost_per_month: 3.00
      input_cost_per_1k: 0.00015
      output_cost_per_1k: 0.0006
  summarizer:
    enabled: true
    model: "gpt-5-nano-2025-08-07"
    base_url: "https://api.openai.com/v1"
    temperature: null
    max_completion_tokens: 16384
    max_text_chars: 50000
    cost_control:
      max_cost_per_run: 0.50
      max_cost_per_month: 10.00
      input_cost_per_1k: 0.00005
      output_cost_per_1k: 0.0004

scoring:
  quality:
    w_venue: 0.25
    w_author: 0.20
    w_cite: 0.20
    w_code: 0.15
    w_fresh: 0.20
  venue_tiers: { ... }
```

**Removed from config:**
- `scoring.alpha` — no relevance/quality blend
- `scoring.relevance` — entire section
- `llm.use_full_text` — always full text now
- Single `llm` block replaced by `llm.filter` + `llm.summarizer`

### New Module: `filter.py`

**System prompt:**
```
You are a research paper relevance filter. The user is interested in:
{topic.description}

Given a paper's title and abstract, decide if it is relevant to these interests.
Respond with JSON: {"relevant": true/false, "reason": "one sentence explaining why"}
```

**Interface:**
- `filter_papers(papers, config, db)` → `(relevant: list[Paper], rejected: list[FilterResult])`
- `FilterResult` = `(paper, relevant: bool, reason: str)`
- Cost tracked per-run in `llm_usage` table

### Database Changes

**New table: `paper_filter_results`**
- `paper_id` (FK to papers), `run_date`, `relevant` (bool), `reason` (text)
- Indexed on `(paper_id, run_date)`

**Modified table: `scores`**
- `relevance` column removed
- `llm_rank` column added (integer, 1 = best)
- `final` remains (stores llm_rank as float for backward compat)

### Model Changes

- `Scores(quality: float, llm_rank: int, final: float)` — `relevance` removed, `llm_rank` added
- New `FilterResult(paper: Paper, relevant: bool, reason: str)`

### Digest Template Changes

- Rejected papers section at bottom: "Reviewed but not included (N papers)"
- Shows title, authors, 1-sentence rejection reason per paper

### CLI Changes

- New `filter` subcommand for running LLM filtering standalone
- `run` pipeline reordered: fetch → filter → enrich → score → digest
- `score` now computes quality only (no relevance)

## What Gets Removed

- `compute_relevance()` from scoring.py
- `score_paper()` / `score_papers()` — replaced by quality-only scoring
- `config.scoring.alpha` and `config.scoring.relevance`
- `config.llm.use_full_text` toggle
- `SYSTEM_PROMPT_ABSTRACT` becomes fallback only (not a config choice)

## What Stays Unchanged

- All collectors (arXiv, blogs, DBLP)
- Dedup logic
- Enrichment modules (Semantic Scholar, PWC)
- Delivery modules (markdown, telegram)
- PDF extraction module
- Database context manager pattern
- Cost tracking pattern (extended to filter)

## Architecture Diagram

```
                          CHEAP MODEL                    GOOD MODEL
                         ┌───────────┐               ┌──────────────┐
Fetch ─► Dedup ─► Store ─► LLM Filter ─► Enrich ─► ─► Summarize     │
  (arXiv, blogs, DBLP)  │ relevant?  │  (S2, PWC)   │ (full text)  │
                         │ + reason   │     │        │ + Rank       │
                         └─────┬──────┘  Quality     └──────┬───────┘
                               │         Score              │
                          Rejected                     Digest
                          papers ──► DB + digest       (ordered by
                                     footer            LLM rank)
```
