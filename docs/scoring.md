# Scoring & Ranking

Papers go through three stages of evaluation: LLM filtering, quality scoring, and LLM ranking.

## LLM Filter (Relevance)

A cheap LLM (gpt-5-nano) reads each paper's title + abstract and decides binary relevance against `topic.description`. This replaces keyword-based relevance scoring.

- **Fail-open** — budget exhaustion or LLM errors treat the paper as relevant
- Decisions stored in `paper_filter_results` table with reasons
- Rejected papers shown in digest footer
- Cost tracked separately with `filter_` prefix on `run_id`

## Quality Score (0–1)

Measures paper quality from external signals, independent of topic.

```
quality = w_venue  * venue_tier(venue)              # 1.0 / 0.7 / 0.4 / 0.2
        + w_author * min(1, max_hindex / 50)        # Author reputation
        + w_cite   * min(1, log(1 + citations) / 5) # Citation impact
        + w_code   * (1.0 if has_code else 0.0)     # Reproducibility signal
        + w_fresh  * max(0, 1 - age_days / 30)      # Linear decay over 30 days
```

Default weights: venue 0.25, author 0.20, citations 0.20, code 0.15, freshness 0.20 (must sum to 1.0).

## LLM Ranking

After quality scoring, the summarizer LLM ranks the top-N papers. `Scores.llm_rank` stores the result (1 = best, 0 = unranked). This provides a final ordering that combines the LLM's judgment with the quality signal.

## LLM Summarization

When enabled, top-N papers get full-text PDF summaries (abstract fallback) with these fields:

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

Multiple safeguards are built in for both filter and summarizer:

- **Per-run limit** — stops if exceeded mid-run
- **Monthly limit** — refuses to call LLM if monthly budget exhausted
- **Dry-run mode** (`--dry-run`) — runs scoring but skips all LLM calls
- **Graceful degradation** — if any LLM call fails, the paper still appears in the digest without a summary

All token usage is tracked in the `llm_usage` SQLite table and visible via `paperdigest stats`.

## Enrichment Sources

### Semantic Scholar

Queries the [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/) for each paper:

- Citation count, author h-indices, venue, open access PDF URL
- Rate limits: 0.5s with API key, 3.5s without; 60s backoff on 429

### Papers with Code

Uses the [PWC links dump](https://paperswithcode.com/about) (~500MB gzipped JSON) for instant local lookups of code repository URLs and official flags. Run `paperdigest init` to download.
