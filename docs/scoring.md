# Scoring & Ranking

Papers go through two stages of LLM evaluation: scoring and ranking.

## LLM Scoring (0–1)

A cheap LLM (gpt-5-nano) reads each paper's title + abstract and assigns a score from 0.0 to 1.0:

| Score | Meaning |
|-------|---------|
| 0.0 | Completely irrelevant to the topic |
| 0.3 | Tangentially related but not directly useful |
| 0.5 | Somewhat relevant, average quality |
| 0.7 | Relevant paper with solid methodology |
| 1.0 | Highly relevant, outstanding paper from respected authors/institution with strong real-world results |

The LLM considers both **topical relevance** (against `topic.description`) and **paper quality** (methodology, authors, results).

- **No rejection** — all papers are scored, sorted by score DESC, and top_n are taken
- **Score threshold** — papers below `digest.score_threshold` (default 0.4) appear in the digest but are not sent to the summarizer LLM, saving costs
- **Fail-open** — budget exhaustion or LLM errors assign a neutral score of 0.5
- Scores stored in `paper_filter_results` table (with `score` column) and `scores` table (`quality` column)
- Cost tracked separately with `filter_` prefix on `run_id`
- Scores are cached — repeated runs skip already-scored papers

## LLM Ranking

After scoring, the summarizer LLM ranks the top-N papers. `Scores.llm_rank` stores the result (1 = best, 0 = unranked). This provides a final ordering that combines the LLM's judgment with the quality signal.

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

Multiple safeguards are built in for both scorer and summarizer:

- **Per-run limit** — stops if exceeded mid-run
- **Monthly limit** — refuses to call LLM if monthly budget exhausted
- **Dry-run mode** (`--dry-run`) — skips all LLM calls
- **Graceful degradation** — if any LLM call fails, the paper still appears in the digest without a summary

All token usage is tracked in the `llm_usage` SQLite table and visible via `paperdigest stats`.

## Enrichment Sources

### Papers with Code

Uses the [PWC links dump](https://paperswithcode.com/about) (~500MB gzipped JSON) for instant local lookups of code repository URLs and official flags. Run `paperdigest init` to download.
