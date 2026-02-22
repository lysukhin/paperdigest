"""Tests for Telegram delivery formatting."""

from __future__ import annotations

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
