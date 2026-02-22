"""Tests for the database layer."""

from datetime import datetime, timezone

import pytest

from paperdigest.db import Database
from paperdigest.models import Paper, Scores


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
