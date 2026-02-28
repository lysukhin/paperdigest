"""Tests for Telegram delivery formatting."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest
import requests

from paperdigest.config import Config, TopicConfig, WebConfig, DeliveryConfig, TelegramDeliveryConfig
from paperdigest.delivery.telegram import _format_telegram_message, _escape_markdown
from paperdigest.models import Digest, DigestEntry, Paper, Scores, Summary


def _make_config(public_url: str | None = None) -> Config:
    """Create a minimal config for testing."""
    return Config(
        topic=TopicConfig(name="Test Topic", primary_keywords=["test"]),
        web=WebConfig(public_url=public_url),
    )


def _make_entry(rank: int, title: str = "Test Paper", one_liner: str = "",
                arxiv_id: str = "2401.00001", authors: list[str] | None = None,
                affiliations: str = "") -> DigestEntry:
    """Create a digest entry for testing."""
    paper = Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="Abstract text",
        authors=authors or ["Author One", "Author Two"],
        published=datetime(2026, 2, 20, tzinfo=timezone.utc),
    )
    scores = Scores(quality=0.75, llm_rank=rank)
    summary = Summary(one_liner=one_liner, affiliations=affiliations) if (one_liner or affiliations) else None
    return DigestEntry(paper=paper, scores=scores, rank=rank, summary=summary)


def _make_digest(n_entries: int = 5, with_summaries: bool = True, number: int = 42) -> Digest:
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
        number=number,
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

    def test_title_truncated_at_100_chars(self):
        """Long titles are truncated at 100 characters."""
        entry = _make_entry(rank=1, title="A" * 120)
        digest = Digest(
            date=datetime(2026, 2, 22, tzinfo=timezone.utc),
            topic_name="Test",
            entries=[entry],
            total_collected=10,
            total_new=1,
        )
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        # 100 A's escaped, not 120
        assert "A" * 101 not in msg

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
                assert markup["inline_keyboard"][0][0]["url"] == "https://digest.example.com/digest/42"

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
            with patch("paperdigest.delivery.telegram.requests.post", side_effect=requests.RequestException("Network error")):
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

    def test_falls_back_to_text_link_on_button_rejection(self):
        """When Telegram rejects the inline button (e.g. localhost), retry with text link."""
        digest = _make_digest(n_entries=1)
        config = _make_config(public_url="http://localhost:8000")

        # First call: 400 error (button rejected). Second call: 200 (text link fallback).
        error_resp = MagicMock(status_code=400)
        error = requests.HTTPError(response=error_resp)
        ok_resp = MagicMock(status_code=200)
        ok_resp.raise_for_status = MagicMock()

        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "fake-token", "TELEGRAM_CHAT_ID": "12345"}):
            with patch("paperdigest.delivery.telegram.requests.post") as mock_post:
                mock_post.side_effect = [error, ok_resp]

                from paperdigest.delivery.telegram import deliver_telegram
                result = deliver_telegram(digest, config)

                assert result is True
                assert mock_post.call_count == 2
                # Second call should have no reply_markup but text link in message
                second_payload = mock_post.call_args_list[1].kwargs.get("json") or mock_post.call_args_list[1][1].get("json")
                assert "reply_markup" not in second_payload
                assert "View Full Digest" in second_payload["text"]


class TestTelegramAuthors:
    def test_shows_first_author_et_al(self):
        """Shows first author + et al. when >2 authors."""
        entry = _make_entry(
            rank=1, title="Paper A", one_liner="Summary",
            authors=["Alice Smith", "Bob Jones", "Charlie Brown"],
            affiliations="MIT, Google",
        )
        digest = Digest(
            date=datetime(2026, 2, 22, tzinfo=timezone.utc),
            topic_name="Test", entries=[entry],
            total_collected=10, total_new=1,
        )
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "Alice Smith" in msg
        assert "et al" in msg

    def test_shows_two_authors_without_et_al(self):
        """Shows both authors when exactly 2."""
        entry = _make_entry(
            rank=1, title="Paper A", one_liner="Summary",
            authors=["Alice Smith", "Bob Jones"],
            affiliations="Stanford",
        )
        digest = Digest(
            date=datetime(2026, 2, 22, tzinfo=timezone.utc),
            topic_name="Test", entries=[entry],
            total_collected=10, total_new=1,
        )
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "Alice Smith" in msg
        assert "Bob Jones" in msg
        assert "et al" not in msg

    def test_shows_affiliations(self):
        """Affiliations from summary are displayed."""
        entry = _make_entry(
            rank=1, title="Paper A", one_liner="Summary",
            affiliations="MIT, Google DeepMind",
        )
        digest = Digest(
            date=datetime(2026, 2, 22, tzinfo=timezone.utc),
            topic_name="Test", entries=[entry],
            total_collected=10, total_new=1,
        )
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "MIT" in msg
        assert "Google DeepMind" in msg

    def test_no_affiliations_shows_authors_only(self):
        """When no affiliations, just shows authors."""
        entry = _make_entry(
            rank=1, title="Paper A", one_liner="Summary",
            authors=["Alice Smith", "Bob Jones"],
            affiliations="",
        )
        digest = Digest(
            date=datetime(2026, 2, 22, tzinfo=timezone.utc),
            topic_name="Test", entries=[entry],
            total_collected=10, total_new=1,
        )
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "Alice Smith" in msg

    def test_no_summary_shows_authors_from_paper(self):
        """When no summary (no affiliations), shows paper authors."""
        entry = _make_entry(
            rank=1, title="Paper A",
            authors=["Alice Smith", "Bob Jones", "Charlie Brown"],
        )
        digest = Digest(
            date=datetime(2026, 2, 22, tzinfo=timezone.utc),
            topic_name="Test", entries=[entry],
            total_collected=10, total_new=1,
        )
        config = _make_config()
        msg = _format_telegram_message(digest, config)
        assert "Alice Smith" in msg
