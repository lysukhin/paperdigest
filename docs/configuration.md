# Configuration Reference

Everything is controlled by `config.yaml`. Below is the full structure.

## Topic

```yaml
topic:
  name: "VLM/VLA for Autonomous Driving"
  description: >                          # Natural language description for LLM filter
    Papers about vision-language models (VLMs), vision-language-action models (VLAs),
    and multimodal foundation models applied to autonomous driving...
  primary_keywords:                       # Used for arXiv query building
    - "autonomous driving"
    - "motion planning"
  secondary_keywords:                     # Optional, used for arXiv queries
    - "end-to-end"
    - "vision-language model"
  benchmarks:                             # Optional
    - "nuScenes"
    - "CARLA"
  arxiv_categories:                       # Filters arXiv queries
    - "cs.CV"
    - "cs.AI"
    - "cs.RO"
    - "cs.LG"
```

`topic.description` is the key field — the LLM filter uses it to decide relevance. Keywords are only used for constructing arXiv API queries.

## Collection

```yaml
collection:
  lookback_days: 7           # How far back to search
  max_results: 200           # Max papers per arXiv query
  blogs:
    enabled: false
    sources:
      - nvidia
      - waymo
      # - wayve             # currently blocked by WAF
  conferences:
    enabled: false
    years_back: 2
    venues:
      - CVPR
      - ICCV
      - ECCV
      # NeurIPS excluded — DBLP API returns 500 for this venue
```

## Scoring

Quality scoring only — relevance is handled by the LLM filter.

```yaml
scoring:
  quality:
    w_venue: 0.25            # Venue tier weight
    w_author: 0.20           # Author h-index weight
    w_cite: 0.20             # Citation count weight
    w_code: 0.15             # Code availability weight
    w_fresh: 0.20            # Recency weight
                             # Weights must sum to 1.0

  venue_tiers:
    tier1: [NeurIPS, ICML, ICLR, CVPR, ECCV, ICCV, AAAI, IJCAI, RSS, CoRL]    # → 1.0
    tier2: [IROS, ICRA, WACV, BMVC, ACCV, AISTATS, UAI]                         # → 0.7
    tier3: [IV, ITSC, T-ITS, RA-L, T-RO]                                        # → 0.4
    # Everything else → 0.2
```

## LLM

Two separate LLM configs — a cheap model for filtering and a capable model for summarization.

```yaml
llm:
  filter:
    enabled: true
    model: "gpt-5-nano-2025-08-07"
    base_url: "https://api.openai.com/v1"
    temperature:                           # null = omit from API call (for reasoning models)
    max_completion_tokens: 1024
    cost_control:
      max_cost_per_run: 0.10
      max_cost_per_month: 3.00
      input_cost_per_1k: 0.00005
      output_cost_per_1k: 0.0004

  summarizer:
    enabled: true
    model: "gpt-5-mini-2025-08-07"
    base_url: "https://api.openai.com/v1"
    temperature:                           # null = omit from API call
    max_completion_tokens: 16384
    max_text_chars: 50000                  # Max PDF text chars sent to LLM
    language: "English"                    # Summary output language
    cost_control:
      max_cost_per_run: 0.50
      max_cost_per_month: 10.00
      input_cost_per_1k: 0.00025
      output_cost_per_1k: 0.002
```

The `base_url` works with any OpenAI-compatible API (OpenAI, Ollama, OpenRouter, etc.).

## Digest & Delivery

```yaml
digest:
  top_n: 20                  # Papers in the digest
  summarize_top_n: 10        # Papers to send to summarizer LLM
  output_dir: "data/digests"

delivery:
  markdown:
    enabled: true            # Always-on local file archive
  telegram:
    enabled: false           # Set true + env vars to enable
```

## Database & PWC

```yaml
database:
  path: "data/papers.db"

pwc:
  links_path: "data/pwc_links.json"
```

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `LLM_API_KEY` | If any LLM enabled | OpenAI or compatible API key |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Faster Semantic Scholar rate limits (0.5s vs 3.5s) |
| `TELEGRAM_BOT_TOKEN` | If Telegram enabled | From [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | If Telegram enabled | Target channel or group ID |

Copy `.env.example` to `.env` and fill in values. The `.env` file is loaded automatically from the same directory as `config.yaml`. Environment variables take precedence over `.env` values.
