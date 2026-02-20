"""Tests for the dedup module."""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from paperdigest.db import Database
from paperdigest.dedup import dedup_papers, normalize_title, titles_match
from paperdigest.models import Paper


def _make_paper(arxiv_id="2401.00001", title="A Test Paper", doi=None) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="Abstract text",
        authors=["Author One"],
        published=datetime.now(timezone.utc),
        doi=doi,
    )


def _make_db() -> Database:
    """Create an in-memory-like temp DB."""
    tmp = tempfile.mktemp(suffix=".db")
    db = Database(tmp)
    db.init_schema()
    return db


class TestTitleNormalization:
    def test_lowercases(self):
        assert normalize_title("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_title("  hello   world  ") == "hello world"


class TestTitleMatching:
    def test_exact_match(self):
        assert titles_match("Hello World", "Hello World")

    def test_case_insensitive(self):
        assert titles_match("Hello World", "hello world")

    def test_fuzzy_match(self):
        assert titles_match(
            "Vision Language Models for Autonomous Driving",
            "Vision Language Model for Autonomous Driving",
        )

    def test_different_titles(self):
        assert not titles_match(
            "Vision Language Models for Autonomous Driving",
            "Protein Folding with Deep Learning",
        )


class TestDedup:
    def test_exact_arxiv_id_dedup(self):
        db = _make_db()
        p1 = _make_paper("2401.00001")
        p2 = _make_paper("2401.00001")
        result = dedup_papers([p1, p2], db)
        assert len(result) == 1

    def test_existing_paper_filtered(self):
        db = _make_db()
        existing = _make_paper("2401.00001")
        db.upsert_paper(existing)
        new = _make_paper("2401.00001")
        result = dedup_papers([new], db)
        assert len(result) == 0

    def test_doi_dedup(self):
        db = _make_db()
        existing = _make_paper("2401.00001", doi="10.1234/test")
        db.upsert_paper(existing)
        new = _make_paper("2401.99999", doi="10.1234/test")
        result = dedup_papers([new], db)
        assert len(result) == 0

    def test_fuzzy_title_within_batch(self):
        db = _make_db()
        p1 = _make_paper("2401.00001", title="Vision Language Models for Autonomous Driving: A Survey")
        p2 = _make_paper("2401.00002", title="Vision Language Model for Autonomous Driving: A Survey")
        result = dedup_papers([p1, p2], db)
        assert len(result) == 1

    def test_different_papers_kept(self):
        db = _make_db()
        p1 = _make_paper("2401.00001", title="Paper About VLM")
        p2 = _make_paper("2401.00002", title="Paper About Protein Folding")
        result = dedup_papers([p1, p2], db)
        assert len(result) == 2
