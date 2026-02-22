# Telegram Digest Notifications with Web Linking

**Date:** 2026-02-22
**Status:** Approved

## Goal

Telegram becomes a compact notification channel: short paper headlines + inline button linking to the full web digest. Push-only, no bot commands.

## Configuration

Add `public_url` to the web config section:

```yaml
web:
  host: 127.0.0.1
  port: 8000
  public_url: https://digest.example.com  # optional, enables Telegram web link button
```

- `public_url: str | None = None` in `WebConfig` dataclass
- When set, Telegram messages include an inline keyboard button to `{public_url}/digest/{date}`
- When `None`, message sent without the button (graceful degradation)

## Telegram Message Format

Top 5 papers, rank + title + one-liner. Single message, well within 4096 chars.

```
Paper Digest: VLM/VLA for AD
2026-02-22 | 45 collected, 12 ranked

1. Multi-Modal Foundation Model for Autonomous Driving
   > Unified VLM achieving SOTA on nuScenes planning

2. LiDAR-Camera Fusion via Vision-Language Alignment
   > Novel cross-modal attention with 15% mAP improvement

3. End-to-End Driving with Chain-of-Thought Reasoning
   > Language-guided planning outperforms IL baselines

4. Real-Time VLA Policy for Urban Navigation
   > 20Hz inference on edge GPU, tested on real vehicle

5. Scaling Laws for Vision-Language Driving Models
   > Empirical study across 7 model scales
```

With an inline keyboard button below:

```
[ View Full Digest ]  -> {public_url}/digest/{date}
```

### Implementation Details

- `reply_markup` parameter on `sendMessage` POST with `InlineKeyboardMarkup`
- Button only present when `public_url` is configured
- No new dependencies (uses existing `requests` + `json`)
- Top 5 hardcoded (reasonable default; the web has the full list)
- Titles truncated at 80 chars (existing behavior, kept)
- One-liners from `Summary.one_liner` (omitted if no summary)

## Cron Setup

Document in README, no code changes:

```
0 9 * * * /path/to/venv/bin/python -m paperdigest run --config /path/to/config.yaml >> /path/to/data/cron.log 2>&1
```

Notes: full path to venv python, `.env` accessible, log rotation optional.

## Scope Boundaries

**In scope:**
- `config.py` — add `public_url` to `WebConfig`
- `delivery/telegram.py` — rewrite message format + add inline button
- `config.yaml` — add `public_url` field (commented out)
- README — cron setup section
- Tests — message formatting, button presence/absence, config parsing

**Out of scope:**
- Web dashboard changes (already serves `/digest/{date}`)
- Bot commands / webhook / polling
- Message splitting across multiple messages
- Digest pipeline changes
- VPS / reverse proxy setup (separate PR)

## Test Plan

- Message formatting: correct compact layout with 5 papers
- Inline button: present when `public_url` set, absent when `None`
- Title truncation at 80 chars
- Papers without summaries: rank + title only, no one-liner line
- Config: `public_url` parsed as string, defaults to `None`
- Graceful degradation: Telegram send failure doesn't break pipeline (existing behavior, verify preserved)
