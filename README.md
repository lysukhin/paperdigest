# paperdigest

Automated research paper digest system. Fetches papers from arXiv, conference proceedings (DBLP), and lab blogs (NVIDIA, Waymo), filters for relevance using an LLM, enriches with citation and code data, scores by quality, generates full-text LLM summaries with ranking, and delivers digests as Markdown, Telegram messages, or a web dashboard.

Ships configured for **VLM/VLA for Autonomous Driving** but works for any research topic via `config.yaml`.

## How It Works

```
arXiv + Blogs + DBLP → Dedup → SQLite → LLM Filter → Semantic Scholar + PWC → Quality Score → [LLM Summary + Rank] → Markdown / Telegram / Web
       (fetch)         (batch)  (store)   (filter)          (enrich)           (score)          (summarize)            (deliver)
```

1. **Collect** — Query arXiv by keywords/categories; optionally scrape NVIDIA/Waymo blogs and DBLP proceedings
2. **Dedup** — Remove duplicates by arXiv ID, DOI, fuzzy title matching (85% threshold), across batch and DB
3. **Filter** — LLM reads title + abstract against `topic.description` and decides relevance (fail-open on errors)
4. **Enrich** — Pull citations, h-indices, venue, OA PDFs from Semantic Scholar; code links from Papers with Code
5. **Score** — Quality score from venue tier, citations, author reputation, code availability, freshness
6. **Summarize** *(optional)* — Fetch full-text PDF, send to LLM for structured summary; LLM also ranks the top papers
7. **Deliver** — Write Markdown digest; push compact top-5 notification to Telegram (with link to full web digest); serve via web dashboard

## Quick Start

```bash
pip install -e .
python -m paperdigest init --skip-pwc
python -m paperdigest run
```

Output lands in `data/digests/digest_YYYY-MM-DD.md`.

## Deploy to VPS (Docker)

Requires Docker and Docker Compose on your VPS.

```bash
# 1. Clone
git clone https://github.com/lysukhin/paperdigest && cd paperdigest

# 2. Run interactive setup (generates config.yaml, .env, Caddyfile, crontab)
docker compose run --rm web paperdigest setup

# 3. Start everything
docker compose up -d
```

This gives you:
- **Web dashboard** with auto-HTTPS (set your domain during setup)
- **Scheduled digests** via cron (default: daily 9:00 UTC)
- **Telegram notifications** (optional, configured during setup)

See `docker compose logs -f` to watch progress.

## Installation

Requires **Python 3.9+**.

```bash
pip install -e .              # core (no LLM)
pip install -e ".[llm]"       # with LLM summarization
pip install -e ".[web]"       # with web dashboard
pip install -e ".[dev]"       # with dev/test tools
```

## CLI Reference

All commands accept `--config <path>` (default: `config.yaml`). Global `-v` flag enables debug logging (before subcommand).

```
python -m paperdigest [-v] <command> [options]
./pd <command>                                    # shorthand wrapper
```

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `run` | Full pipeline: fetch → digest | |
| `fetch` | Collect papers from arXiv (+ blogs/DBLP if enabled) | |
| `filter` | Run LLM relevance filtering | |
| `enrich` | Add Semantic Scholar + PWC data | |
| `score` | Compute quality scores | |
| `digest` | Generate and deliver digest (filter→enrich→score→summarize→rank) | `--dry-run` |
| `init` | Create database, download PWC links | `--skip-pwc` |
| `serve` | Start web dashboard (localhost:8000) | |
| `stats` | Show DB and LLM usage statistics | |
| `clean` | Clean old data | |

## Configuration

Controlled by `config.yaml`. See **[docs/configuration.md](docs/configuration.md)** for the full reference with all YAML blocks.

Key sections: topic description + keywords, collection sources, quality weights + venue tiers, LLM filter + summarizer config, delivery channels, `web.public_url` for Telegram digest links.

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `LLM_API_KEY` | If any LLM enabled | OpenAI or compatible API key |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Faster Semantic Scholar rate limits |
| `OPENAI_ADMIN_KEY` | No | Real OpenAI usage/costs on dashboard and `stats` |
| `TELEGRAM_BOT_TOKEN` | If Telegram enabled | From [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | If Telegram enabled | Target channel or group ID |

Copy `.env.example` to `.env` and fill in values.

## Scoring & LLM

A cheap LLM filters papers for topic relevance. Survivors get a quality score (venue, citations, h-index, code, freshness), then a capable LLM generates full-text summaries and ranks the top papers.

See **[docs/scoring.md](docs/scoring.md)** for formulas, enrichment sources, and cost controls.

## Adapting to a Different Topic

1. Change `topic.name`, `topic.description`, keywords, and benchmarks to your field
2. Update `arxiv_categories` (see [arXiv taxonomy](https://arxiv.org/category_taxonomy))
3. Adjust `venue_tiers` for your field's top conferences
4. Run `python -m paperdigest init --skip-pwc && python -m paperdigest run`

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Scheduled Runs (Cron)

To run digests daily, add a crontab entry:

```bash
# Run daily at 9:00 AM
0 9 * * * /path/to/venv/bin/python -m paperdigest run --config /path/to/config.yaml >> /path/to/data/cron.log 2>&1
```

Notes:
- Use the full path to your virtualenv's Python to ensure correct dependencies
- The `--config` flag accepts an absolute path to your config file
- `.env` must be in the same directory as `config.yaml`

## Roadmap

See **[docs/roadmap.md](docs/roadmap.md)** for planned features (additional collectors, enrichment signals, advanced scoring, delivery channels, infrastructure).

## License

MIT
