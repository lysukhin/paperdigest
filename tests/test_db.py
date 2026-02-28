"""Tests for the database layer."""

import sqlite3
from datetime import datetime, timezone

import pytest

from paperdigest.db import Database
from paperdigest.models import Paper, Scores, Summary


def _make_paper(
    arxiv_id="2401.00001",
    title="A Paper",
    abstract="An abstract about autonomous driving",
    published=None,
) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=["Alice", "Bob"],
        published=published or datetime(2024, 1, 15, tzinfo=timezone.utc),
    )


def _setup_db(tmp_path):
    """Create a fresh database with schema initialized."""
    db = Database(tmp_path / "test.db")
    db.init_schema()
    return db


class TestFilterResults:
    def test_upsert_filter_result_relevant(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        db.upsert_filter_result(paper_id, relevant=True, reason="Matches topic")

        results = db.get_filter_results()
        assert len(results) == 1
        assert results[0]["paper_id"] == paper_id
        assert results[0]["relevant"] == 1
        assert results[0]["reason"] == "Matches topic"
        db.close()

    def test_upsert_filter_result_not_relevant(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        db.upsert_filter_result(paper_id, relevant=False, reason="Off topic")

        results = db.get_filter_results()
        assert len(results) == 1
        assert results[0]["relevant"] == 0
        assert results[0]["reason"] == "Off topic"
        db.close()

    def test_get_filter_results_by_date(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        db.upsert_filter_result(paper_id, relevant=True, reason="Good paper")

        # The run_date is set to date('now') which is today
        today = datetime.now().strftime("%Y-%m-%d")
        results = db.get_filter_results(run_date=today)
        assert len(results) == 1

        results_other = db.get_filter_results(run_date="2020-01-01")
        assert len(results_other) == 0
        db.close()

    def test_get_rejected_papers(self, tmp_path):
        db = _setup_db(tmp_path)

        p1_id = db.upsert_paper(_make_paper(arxiv_id="2401.00001", title="Relevant Paper"))
        p2_id = db.upsert_paper(_make_paper(arxiv_id="2401.00002", title="Irrelevant Paper"))
        p3_id = db.upsert_paper(_make_paper(arxiv_id="2401.00003", title="Another Irrelevant"))

        db.upsert_filter_result(p1_id, relevant=True, reason="Matches topic")
        db.upsert_filter_result(p2_id, relevant=False, reason="Not about AD")
        db.upsert_filter_result(p3_id, relevant=False, reason="Wrong domain")

        rejected = db.get_rejected_papers()
        assert len(rejected) == 2
        titles = [paper.title for paper, reason in rejected]
        assert "Irrelevant Paper" in titles
        assert "Another Irrelevant" in titles
        reasons = [reason for paper, reason in rejected]
        assert "Not about AD" in reasons
        assert "Wrong domain" in reasons
        db.close()

    def test_get_rejected_papers_by_date(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        db.upsert_filter_result(paper_id, relevant=False, reason="Off topic")

        today = datetime.now().strftime("%Y-%m-%d")
        rejected = db.get_rejected_papers(run_date=today)
        assert len(rejected) == 1
        assert rejected[0][1] == "Off topic"

        rejected_other = db.get_rejected_papers(run_date="2020-01-01")
        assert len(rejected_other) == 0
        db.close()

    def test_multiple_filter_results_same_paper(self, tmp_path):
        """A paper can have filter results from multiple runs."""
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())

        # Insert two results (simulating different runs on same date)
        db.upsert_filter_result(paper_id, relevant=True, reason="First run")
        db.upsert_filter_result(paper_id, relevant=False, reason="Second run")

        results = db.get_filter_results()
        assert len(results) == 2
        db.close()

    def test_get_latest_filter_result(self, tmp_path):
        """Returns the most recent filter result for a paper."""
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        db.upsert_filter_result(paper_id, relevant=True, reason="First")
        db.upsert_filter_result(paper_id, relevant=False, reason="Second")
        result = db.get_latest_filter_result(paper_id)
        assert result is not None
        assert result["reason"] == "Second"
        db.close()

    def test_get_latest_filter_result_no_results(self, tmp_path):
        """Returns None if paper has never been filtered."""
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        result = db.get_latest_filter_result(paper_id)
        assert result is None
        db.close()


class TestUpdatedScores:
    def test_upsert_scores_with_llm_rank(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        scores = Scores(quality=0.85, llm_rank=3)
        db.upsert_scores(paper_id, scores)

        results = db.get_top_scored_papers(limit=10)
        assert len(results) == 1
        paper, s = results[0]
        assert s.quality == pytest.approx(0.85)
        assert s.llm_rank == 3
        db.close()

    def test_upsert_scores_update_existing(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())

        db.upsert_scores(paper_id, Scores(quality=0.5, llm_rank=5))
        db.upsert_scores(paper_id, Scores(quality=0.9, llm_rank=1))

        results = db.get_top_scored_papers(limit=10)
        assert len(results) == 1
        _, s = results[0]
        assert s.quality == pytest.approx(0.9)
        assert s.llm_rank == 1
        db.close()

    def test_top_scored_ordering_by_llm_rank_then_quality(self, tmp_path):
        """Papers should be ordered by llm_rank ASC, then quality DESC."""
        db = _setup_db(tmp_path)

        # Create papers with different ranks and qualities
        p1_id = db.upsert_paper(_make_paper(arxiv_id="2401.00001", title="Rank 1"))
        p2_id = db.upsert_paper(_make_paper(arxiv_id="2401.00002", title="Rank 2"))
        p3_id = db.upsert_paper(_make_paper(arxiv_id="2401.00003", title="Rank 3 High Q"))
        p4_id = db.upsert_paper(_make_paper(arxiv_id="2401.00004", title="Rank 3 Low Q"))

        db.upsert_scores(p1_id, Scores(quality=0.7, llm_rank=1))
        db.upsert_scores(p2_id, Scores(quality=0.9, llm_rank=2))
        db.upsert_scores(p3_id, Scores(quality=0.8, llm_rank=3))
        db.upsert_scores(p4_id, Scores(quality=0.5, llm_rank=3))

        results = db.get_top_scored_papers(limit=10)
        assert len(results) == 4

        # llm_rank=1 first
        assert results[0][0].title == "Rank 1"
        assert results[0][1].llm_rank == 1

        # llm_rank=2 second
        assert results[1][0].title == "Rank 2"
        assert results[1][1].llm_rank == 2

        # llm_rank=3 with higher quality third
        assert results[2][0].title == "Rank 3 High Q"
        assert results[2][1].quality == pytest.approx(0.8)

        # llm_rank=3 with lower quality last
        assert results[3][0].title == "Rank 3 Low Q"
        assert results[3][1].quality == pytest.approx(0.5)
        db.close()

    def test_top_scored_limit(self, tmp_path):
        db = _setup_db(tmp_path)

        for i in range(5):
            pid = db.upsert_paper(
                _make_paper(arxiv_id=f"2401.{i:05d}", title=f"Paper {i}")
            )
            db.upsert_scores(pid, Scores(quality=0.5 + i * 0.1, llm_rank=i + 1))

        results = db.get_top_scored_papers(limit=3)
        assert len(results) == 3
        # First result should be llm_rank=1
        assert results[0][1].llm_rank == 1
        db.close()

    def test_scores_default_llm_rank_zero(self, tmp_path):
        """Unranked papers have llm_rank=0."""
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        scores = Scores(quality=0.6)
        db.upsert_scores(paper_id, scores)

        results = db.get_top_scored_papers(limit=10)
        assert len(results) == 1
        _, s = results[0]
        assert s.llm_rank == 0
        assert s.quality == pytest.approx(0.6)
        db.close()


class TestDigestedAt:
    def test_new_paper_has_no_digested_at(self, tmp_path):
        """A freshly inserted paper should appear in undigested papers."""
        db = _setup_db(tmp_path)
        db.upsert_paper(_make_paper())

        undigested = db.get_undigested_papers()
        assert len(undigested) == 1
        assert undigested[0].title == "A Paper"
        db.close()

    def test_mark_papers_digested(self, tmp_path):
        """Marking papers as digested removes them from undigested list."""
        db = _setup_db(tmp_path)
        id1 = db.upsert_paper(_make_paper(arxiv_id="2401.00001", title="Paper 1"))
        id2 = db.upsert_paper(_make_paper(arxiv_id="2401.00002", title="Paper 2"))
        id3 = db.upsert_paper(_make_paper(arxiv_id="2401.00003", title="Paper 3"))

        db.mark_papers_digested([id1, id2])

        undigested = db.get_undigested_papers()
        assert len(undigested) == 1
        assert undigested[0].title == "Paper 3"
        db.close()

    def test_get_undigested_papers_returns_newest_first(self, tmp_path):
        """Undigested papers should be ordered by published DESC."""
        db = _setup_db(tmp_path)
        db.upsert_paper(
            _make_paper(
                arxiv_id="2401.00001",
                title="Old Paper",
                published=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        )
        db.upsert_paper(
            _make_paper(
                arxiv_id="2401.00003",
                title="Newest Paper",
                published=datetime(2024, 3, 1, tzinfo=timezone.utc),
            )
        )
        db.upsert_paper(
            _make_paper(
                arxiv_id="2401.00002",
                title="Middle Paper",
                published=datetime(2024, 2, 1, tzinfo=timezone.utc),
            )
        )

        undigested = db.get_undigested_papers()
        assert len(undigested) == 3
        assert undigested[0].title == "Newest Paper"
        assert undigested[1].title == "Middle Paper"
        assert undigested[2].title == "Old Paper"
        db.close()

    def test_mark_papers_digested_empty_list(self, tmp_path):
        """Calling mark_papers_digested with empty list is a no-op."""
        db = _setup_db(tmp_path)
        db.upsert_paper(_make_paper())

        db.mark_papers_digested([])

        undigested = db.get_undigested_papers()
        assert len(undigested) == 1
        db.close()


class TestMigration:
    def test_migrate_old_scores_table(self, tmp_path):
        """Test migration from old scores schema (relevance, quality, final) to new (quality, llm_rank)."""
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys=ON")
        # Create old schema
        conn.executescript("""
            CREATE TABLE papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT NOT NULL,
                published TEXT NOT NULL,
                updated TEXT, doi TEXT, categories TEXT, pdf_url TEXT,
                citations INTEGER, max_hindex INTEGER, venue TEXT,
                oa_pdf_url TEXT, code_url TEXT, code_official INTEGER DEFAULT 0,
                created_at TEXT, updated_at TEXT
            );
            CREATE TABLE scores (
                paper_id INTEGER PRIMARY KEY REFERENCES papers(id),
                relevance REAL NOT NULL,
                quality REAL NOT NULL,
                final REAL NOT NULL,
                scored_at TEXT DEFAULT (datetime('now'))
            );
        """)
        # Insert test data
        conn.execute(
            "INSERT INTO papers (arxiv_id, title, abstract, authors, published) "
            "VALUES ('2401.00001', 'Test', 'Abstract', '[]', '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO scores (paper_id, relevance, quality, final) "
            "VALUES (1, 0.8, 0.6, 0.72)"
        )
        conn.commit()
        conn.close()

        # Open with Database (which calls init_schema -> migrate)
        with Database(db_path) as db:
            db.init_schema()
            # Verify migration
            row = db.conn.execute("SELECT * FROM scores WHERE paper_id = 1").fetchone()
            assert row["quality"] == pytest.approx(0.6)
            assert row["llm_rank"] == 0
            # Verify old columns are gone
            cursor = db.conn.execute("PRAGMA table_info(scores)")
            columns = {r[1] for r in cursor.fetchall()}
            assert "relevance" not in columns
            assert "final" not in columns
            assert "llm_rank" in columns

    def test_migrate_backfills_digested_at_from_digests_table(self, tmp_path):
        """Migration backfills digested_at for papers in previous digests."""
        db_path = tmp_path / "backfill.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        # Create schema WITHOUT digested_at (pre-migration state)
        conn.executescript("""
            CREATE TABLE papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT NOT NULL,
                published TEXT NOT NULL,
                updated TEXT, doi TEXT, categories TEXT, pdf_url TEXT,
                citations INTEGER, max_hindex INTEGER, venue TEXT,
                oa_pdf_url TEXT, code_url TEXT, code_official INTEGER DEFAULT 0,
                created_at TEXT, updated_at TEXT
            );
            CREATE TABLE scores (
                paper_id INTEGER PRIMARY KEY REFERENCES papers(id),
                quality REAL NOT NULL,
                llm_rank INTEGER NOT NULL DEFAULT 0,
                scored_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE digests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                paper_ids TEXT NOT NULL,
                delivery_status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        # Insert 3 papers
        for i in range(1, 4):
            conn.execute(
                "INSERT INTO papers (arxiv_id, title, abstract, authors, published) "
                f"VALUES ('2401.{i:05d}', 'Paper {i}', 'Abstract', '[]', '2024-01-01')"
            )
        # Paper 1 and 2 were in a previous digest
        conn.execute(
            "INSERT INTO digests (date, paper_ids, delivery_status) VALUES (?, ?, ?)",
            ("2024-01-15", "[1, 2]", "delivered"),
        )
        conn.commit()
        conn.close()

        # Open with Database which runs migration + backfill
        with Database(db_path) as db:
            db.init_schema()
            # Papers 1 and 2 should be marked digested
            undigested = db.get_undigested_papers()
            assert len(undigested) == 1
            assert undigested[0].arxiv_id == "2401.00003"

    def test_migrate_backfills_digest_number(self, tmp_path):
        """Migration assigns sequential digest_number to existing digests."""
        db_path = tmp_path / "digest_num.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        # Create old schema without digest_number
        conn.executescript("""
            CREATE TABLE papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                authors TEXT NOT NULL,
                published TEXT NOT NULL,
                updated TEXT, doi TEXT, categories TEXT, pdf_url TEXT,
                citations INTEGER, max_hindex INTEGER, venue TEXT,
                oa_pdf_url TEXT, code_url TEXT, code_official INTEGER DEFAULT 0,
                created_at TEXT, updated_at TEXT, digested_at TEXT
            );
            CREATE TABLE scores (
                paper_id INTEGER PRIMARY KEY REFERENCES papers(id),
                quality REAL NOT NULL,
                llm_rank INTEGER NOT NULL DEFAULT 0,
                scored_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE digests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                paper_ids TEXT NOT NULL,
                delivery_status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.execute("INSERT INTO digests (date, paper_ids) VALUES ('2024-01-01', '[1]')")
        conn.execute("INSERT INTO digests (date, paper_ids) VALUES ('2024-01-02', '[2]')")
        conn.commit()
        conn.close()

        with Database(db_path) as db:
            db.init_schema()
            rows = db.conn.execute(
                "SELECT digest_number FROM digests ORDER BY digest_number"
            ).fetchall()
            assert [r[0] for r in rows] == [1, 2]

    def test_migration_idempotent(self, tmp_path):
        """Migration should be safe to run multiple times."""
        with Database(tmp_path / "test.db") as db:
            db.init_schema()
            db.init_schema()  # Second call should not fail


class TestDigestNumbering:
    def test_log_digest_returns_digest_number(self, tmp_path):
        db = _setup_db(tmp_path)
        p_id = db.upsert_paper(_make_paper())
        number = db.log_digest([p_id], status="delivered")
        assert number == 1
        db.close()

    def test_digest_numbers_auto_increment(self, tmp_path):
        db = _setup_db(tmp_path)
        p1 = db.upsert_paper(_make_paper(arxiv_id="2401.00001"))
        p2 = db.upsert_paper(_make_paper(arxiv_id="2401.00002"))
        n1 = db.log_digest([p1], status="delivered")
        n2 = db.log_digest([p2], status="delivered")
        assert n1 == 1
        assert n2 == 2
        db.close()

    def test_update_digest_status(self, tmp_path):
        db = _setup_db(tmp_path)
        p_id = db.upsert_paper(_make_paper())
        number = db.log_digest([p_id], status="pending")
        db.update_digest_status(number, "delivered")
        row = db.conn.execute(
            "SELECT delivery_status FROM digests WHERE digest_number = ?", (number,)
        ).fetchone()
        assert row["delivery_status"] == "delivered"
        db.close()

    def test_get_next_digest_number_with_no_digests(self, tmp_path):
        db = _setup_db(tmp_path)
        p_id = db.upsert_paper(_make_paper())
        number = db.log_digest([p_id])
        assert number == 1
        db.close()


class TestSummaryPersistence:
    def test_upsert_and_get_summary(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        summary = Summary(
            one_liner="A novel approach",
            affiliations="MIT",
            method="We propose X",
            data_benchmarks="nuScenes",
            key_results="95% accuracy",
            novelty="First to do Y",
            ad_relevance="Driving perception",
            limitations="Simulation only",
        )
        db.upsert_summary(paper_id, summary)
        result = db.get_summary(paper_id)
        assert result is not None
        assert result.one_liner == "A novel approach"
        assert result.affiliations == "MIT"
        assert result.method == "We propose X"
        assert result.data_benchmarks == "nuScenes"
        assert result.key_results == "95% accuracy"
        assert result.novelty == "First to do Y"
        assert result.ad_relevance == "Driving perception"
        assert result.limitations == "Simulation only"
        db.close()

    def test_get_summary_returns_none_when_missing(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        result = db.get_summary(paper_id)
        assert result is None
        db.close()

    def test_upsert_summary_overwrites(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        db.upsert_summary(paper_id, Summary(one_liner="Old"))
        db.upsert_summary(paper_id, Summary(one_liner="New"))
        result = db.get_summary(paper_id)
        assert result.one_liner == "New"
        db.close()
