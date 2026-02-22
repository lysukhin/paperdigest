# Telegram Digest Notifications Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite Telegram delivery to send compact paper headlines (top 5) with an inline keyboard button linking to the full web digest.

**Architecture:** Add `public_url` to `WebConfig`, rewrite `_format_telegram_message` for a compact format, add `reply_markup` with `InlineKeyboardButton` to the `sendMessage` payload. No new dependencies.

**Tech Stack:** Python dataclasses, requests (existing), Telegram Bot API `sendMessage` with `reply_markup`

---

### Task 1: Add `public_url` to WebConfig

**Files:**
- Modify: `src/paperdigest/config.py:116-118` (WebConfig dataclass)
- Modify: `src/paperdigest/config.py:327-329` (web config construction in `load_config`)
- Test: `tests/test_config.py`

**Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
class TestWebConfig:
    def test_web_public_url_default_none(self, tmp_path):
        """public_url defaults to None when not set."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            }
        })
        config = load_config(path)
        assert config.web.public_url is None

    def test_web_public_url_loaded(self, tmp_path):
        """public_url is loaded from config."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "web": {
                "public_url": "https://digest.example.com",
            },
        })
        config = load_config(path)
        assert config.web.public_url == "https://digest.example.com"

    def test_web_public_url_strips_trailing_slash(self, tmp_path):
        """public_url has trailing slash stripped."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "web": {
                "public_url": "https://digest.example.com/",
            },
        })
        config = load_config(path)
        assert config.web.public_url == "https://digest.example.com"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py::TestWebConfig -v`
Expected: FAIL — `WebConfig` has no `public_url` attribute

**Step 3: Implement**

In `src/paperdigest/config.py`, add `public_url` field to `WebConfig`:

```python
@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    public_url: str | None = None
```

In `load_config`, update the web config construction (around line 327):

```python
        web=WebConfig(
            host=web_raw.get("host", "127.0.0.1"),
            port=web_raw.get("port", 8000),
            public_url=web_raw.get("public_url", "").rstrip("/") or None,
        ),
```

The `rstrip("/") or None` handles: no key → `None`, empty string → `None`, trailing slash → stripped, normal URL → kept.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py::TestWebConfig -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/paperdigest/config.py tests/test_config.py
git commit -m "feat: add public_url to WebConfig for Telegram digest links"
```

---

### Task 2: Rewrite Telegram message formatting

**Files:**
- Modify: `src/paperdigest/delivery/telegram.py:25-44` (`_format_telegram_message`)
- Create: `tests/test_telegram.py`

**Step 1: Write the failing tests**

Create `tests/test_telegram.py`:

```python
"""Tests for Telegram delivery formatting."""

from datetime import datetime, timezone

import pytest

from paperdigest.config import Config, TopicConfig, WebConfig, DeliveryConfig, TelegramDeliveryConfig
from paperdigest.delivery.telegram import _format_telegram_message, _escape_markdown
from paperdigest.models import Digest, DigestEntry, Paper, Scores, Summary


def _make_config(public_url: str | None = None) -> Config:
    """Create a minimal config for testing."""
    return Config(
        topic=TopicConfig(name="Test Topic", primary_keywords=["test"]),
        web=WebConfig(public_url=public_url),
    )


def _make_entry(rank: int, title: str = "Test Paper", one_liner: str = "", arxiv_id: str = "2401.00001") -> DigestEntry:
    """Create a digest entry for testing."""
    paper = Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="Abstract text",
        authors=["Author One", "Author Two"],
        published=datetime(2026, 2, 20, tzinfo=timezone.utc),
    )
    scores = Scores(quality=0.75, llm_rank=rank)
    summary = Summary(one_liner=one_liner) if one_liner else None
    return DigestEntry(paper=paper, scores=scores, rank=rank, summary=summary)


def _make_digest(n_entries: int = 5, with_summaries: bool = True) -> Digest:
    """Create a digest with N entries."""
    entries = []
    for i in range(1, n_entries + 1):
        one_liner = f"Summary for paper {i}" if with_summaries else ""
        entries.append(_make_entry(
            rank=i,
            title=f"Paper Title Number {i}",
            one_liner=one_liner,
            arxiv_id=f"2401.{i:05d}",
        ))
    return Digest(
        date=datetime(2026, 2, 22, tzinfo=timezone.utc),
        topic_name="VLM/VLA for AD",
        entries=entries,
        total_collected=45,
        total_new=12,
    )


class TestTelegramFormatting:
    def test_header_contains_topic_and_date(self):
        """Message starts with topic name and date."""
        digest = _make_digest(n_entries=1)
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "VLM/VLA for AD" in msg
        assert "2026\\-02\\-22" in msg

    def test_header_contains_stats(self):
        """Header shows collected and ranked counts."""
        digest = _make_digest(n_entries=3)
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "45 collected" in msg
        assert "3 ranked" in msg

    def test_shows_top_5_papers(self):
        """Message includes up to 5 paper entries."""
        digest = _make_digest(n_entries=7)
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "Paper Title Number 5" in msg
        assert "Paper Title Number 6" not in msg

    def test_entry_has_rank_and_title(self):
        """Each entry shows rank number and title."""
        digest = _make_digest(n_entries=2, with_summaries=False)
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "1\\." in msg
        assert "Paper Title Number 1" in msg

    def test_entry_with_summary_shows_one_liner(self):
        """Entry with summary shows the one-liner."""
        digest = _make_digest(n_entries=1, with_summaries=True)
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "Summary for paper 1" in msg

    def test_entry_without_summary_no_one_liner(self):
        """Entry without summary omits the one-liner line."""
        digest = _make_digest(n_entries=1, with_summaries=False)
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "Summary for paper" not in msg

    def test_title_truncated_at_80_chars(self):
        """Long titles are truncated."""
        entry = _make_entry(rank=1, title="A" * 100)
        digest = Digest(
            date=datetime(2026, 2, 22, tzinfo=timezone.utc),
            topic_name="Test",
            entries=[entry],
            total_collected=10,
            total_new=1,
        )
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        # 80 A's escaped, not 100
        assert "A" * 81 not in msg

    def test_fewer_than_5_papers_shows_all(self):
        """When digest has fewer than 5 entries, show all."""
        digest = _make_digest(n_entries=2)
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "Paper Title Number 1" in msg
        assert "Paper Title Number 2" in msg

    def test_special_chars_escaped(self):
        """Markdown special characters in titles are escaped."""
        entry = _make_entry(rank=1, title="Test: A (Novel) Approach [v2]")
        digest = Digest(
            date=datetime(2026, 2, 22, tzinfo=timezone.utc),
            topic_name="Test",
            entries=[entry],
            total_collected=10,
            total_new=1,
        )
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        # Parens and brackets should be escaped
        assert "\\(" in msg
        assert "\\[" in msg
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_telegram.py::TestTelegramFormatting -v`
Expected: FAIL — current `_format_telegram_message` shows 10 entries, different format

**Step 3: Rewrite `_format_telegram_message`**

Replace `_format_telegram_message` in `src/paperdigest/delivery/telegram.py`:

```python
TOP_N_TELEGRAM = 5


def _format_telegram_message(digest: Digest, config: Config) -> str:
    """Format digest as a compact Telegram message (MarkdownV2)."""
    topic = _escape_markdown(digest.topic_name)
    date_str = _escape_markdown(digest.date.strftime("%Y-%m-%d"))
    n_ranked = len(digest.entries)

    lines = [
        f"*Paper Digest: {topic}*",
        f"_{date_str}_ \\| {digest.total_collected} collected, {n_ranked} ranked",
        "",
    ]

    for entry in digest.entries[:TOP_N_TELEGRAM]:
        title = _escape_markdown(entry.paper.title[:80])
        lines.append(f"*{entry.rank}\\.*  {title}")
        if entry.summary and entry.summary.one_liner:
            one_liner = _escape_markdown(entry.summary.one_liner)
            lines.append(f"  _\\> {one_liner}_")

    return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_telegram.py::TestTelegramFormatting -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/paperdigest/delivery/telegram.py tests/test_telegram.py
git commit -m "feat: rewrite Telegram message to compact top-5 format"
```

---

### Task 3: Add inline keyboard button to `deliver_telegram`

**Files:**
- Modify: `src/paperdigest/delivery/telegram.py:47-80` (`deliver_telegram`)
- Test: `tests/test_telegram.py`

**Step 1: Write the failing tests**

Add to `tests/test_telegram.py`:

```python
import json
from unittest.mock import patch, MagicMock


class TestTelegramDelivery:
    def test_sends_with_inline_button_when_public_url_set(self):
        """When public_url is configured, payload includes reply_markup."""
        digest = _make_digest(n_entries=2)
        config = _make_config(public_url="https://digest.example.com")

        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake-token", "TELEGRAM_CHAT_ID": "12345"}):
            with patch("paperdigest.delivery.telegram.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200)
                mock_post.return_value.raise_for_status = MagicMock()

                from paperdigest.delivery.telegram import deliver_telegram
                deliver_telegram(digest, config)

                call_kwargs = mock_post.call_args
                payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
                assert "reply_markup" in payload
                markup = json.loads(payload["reply_markup"])
                assert markup["inline_keyboard"][0][0]["url"] == "https://digest.example.com/digest/2026-02-22"

    def test_no_button_when_public_url_not_set(self):
        """When public_url is None, payload has no reply_markup."""
        digest = _make_digest(n_entries=2)
        config = _make_config(public_url=None)

        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake-token", "TELEGRAM_CHAT_ID": "12345"}):
            with patch("paperdigest.delivery.telegram.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200)
                mock_post.return_value.raise_for_status = MagicMock()

                from paperdigest.delivery.telegram import deliver_telegram
                deliver_telegram(digest, config)

                call_kwargs = mock_post.call_args
                payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
                assert "reply_markup" not in payload

    def test_returns_false_when_token_missing(self):
        """Returns False when TELEGRAM_BOT_TOKEN is not set."""
        digest = _make_digest(n_entries=1)
        config = _make_config()

        with patch.dict("os.environ", {}, clear=True):
            # Ensure env vars are not set
            import os
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            os.environ.pop("TELEGRAM_CHAT_ID", None)

            from paperdigest.delivery.telegram import deliver_telegram
            result = deliver_telegram(digest, config)
            assert result is False

    def test_returns_false_on_request_error(self):
        """Returns False on network error, doesn't raise."""
        digest = _make_digest(n_entries=1)
        config = _make_config()

        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake-token", "TELEGRAM_CHAT_ID": "12345"}):
            with patch("paperdigest.delivery.telegram.requests.post", side_effect=Exception("Network error")):
                from paperdigest.delivery.telegram import deliver_telegram
                result = deliver_telegram(digest, config)
                assert result is False

    def test_button_text(self):
        """Inline button has correct display text."""
        digest = _make_digest(n_entries=1)
        config = _make_config(public_url="https://digest.example.com")

        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake-token", "TELEGRAM_CHAT_ID": "12345"}):
            with patch("paperdigest.delivery.telegram.requests.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=200)
                mock_post.return_value.raise_for_status = MagicMock()

                from paperdigest.delivery.telegram import deliver_telegram
                deliver_telegram(digest, config)

                payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
                markup = json.loads(payload["reply_markup"])
                button_text = markup["inline_keyboard"][0][0]["text"]
                assert "View Full Digest" in button_text
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_telegram.py::TestTelegramDelivery -v`
Expected: FAIL — current `deliver_telegram` doesn't send `reply_markup`

**Step 3: Implement**

Add `import json` at the top of `telegram.py`, then rewrite `deliver_telegram`:

```python
def deliver_telegram(digest: Digest, config: Config) -> bool:
    """Send digest via Telegram Bot API."""
    token = config.telegram_bot_token
    chat_id = config.telegram_chat_id

    if not token or not chat_id:
        logger.warning("Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return False

    message = _format_telegram_message(digest, config)

    # Truncate if too long
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[: MAX_MESSAGE_LENGTH - 20] + "\n\n_\\.\\.\\.truncated_"

    url = TELEGRAM_API.format(token=token)
    payload: dict = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True,
    }

    # Add inline button linking to web digest
    public_url = config.web.public_url
    if public_url:
        date_str = digest.date.strftime("%Y-%m-%d")
        digest_url = f"{public_url}/digest/{date_str}"
        payload["reply_markup"] = json.dumps({
            "inline_keyboard": [[
                {"text": "\U0001f4d6 View Full Digest", "url": digest_url}
            ]]
        })

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram message sent successfully")
        return True
    except requests.RequestException as e:
        sanitized = str(e).replace(token, "<REDACTED>")
        logger.error(f"Failed to send Telegram message: {sanitized}")
        return False
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_telegram.py -v`
Expected: PASS (all 14 tests — 9 formatting + 5 delivery)

**Step 5: Commit**

```bash
git add src/paperdigest/delivery/telegram.py tests/test_telegram.py
git commit -m "feat: add inline keyboard button for web digest link"
```

---

### Task 4: Update config.yaml and documentation

**Files:**
- Modify: `config.yaml:124-128` (delivery section — add `public_url` example)
- Modify: `docs/configuration.md` (if it documents web config)
- Modify: `README.md` (add cron setup section)

**Step 1: Add `public_url` to config.yaml**

In `config.yaml`, add a commented-out `public_url` under the `web` section. Since there's no explicit `web:` block in config.yaml currently, add it at the end:

```yaml
web:
  # public_url: https://digest.example.com  # enables "View Full Digest" button in Telegram
```

**Step 2: Add cron setup to README**

Add a "Scheduled Runs" section to README.md with:

```markdown
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
```

**Step 3: Update docs/configuration.md if it has a web section**

Check if `docs/configuration.md` documents the `web:` config block. If so, add `public_url` documentation.

**Step 4: Commit**

```bash
git add config.yaml README.md docs/configuration.md
git commit -m "docs: add public_url config and cron setup instructions"
```

---

### Task 5: Run full test suite and verify

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS, including new `test_telegram.py` and `test_config.py::TestWebConfig`

**Step 2: Manual smoke test (optional)**

If Telegram is configured locally:
```bash
python -m paperdigest digest --dry-run
```
Verify the Telegram message format looks correct.

**Step 3: Final commit if any fixes needed**

Fix any issues found, commit with appropriate message.
