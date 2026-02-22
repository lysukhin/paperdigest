"""Tests for data models."""

from datetime import datetime

from paperdigest.models import Digest, DigestEntry, FilterResult, Paper, Scores, Summary


def _make_paper(**overrides) -> Paper:
    defaults = {
        "arxiv_id": "2401.00001",
        "title": "Test Paper",
        "abstract": "An abstract.",
        "authors": ["Author A"],
        "published": datetime(2024, 1, 1),
    }
    defaults.update(overrides)
    return Paper(**defaults)


class TestScores:
    def test_defaults(self):
        s = Scores()
        assert s.quality == 0.0
        assert s.llm_rank == 0

    def test_custom_values(self):
        s = Scores(quality=0.85, llm_rank=3)
        assert s.quality == 0.85
        assert s.llm_rank == 3

    def test_no_relevance_field(self):
        """Scores no longer has a relevance field."""
        assert not hasattr(Scores(), "relevance")

    def test_no_final_field(self):
        """Scores no longer has a final field."""
        assert not hasattr(Scores(), "final")


class TestFilterResult:
    def test_defaults(self):
        paper = _make_paper()
        fr = FilterResult(paper=paper, relevant=True)
        assert fr.paper is paper
        assert fr.relevant is True
        assert fr.reason == ""

    def test_with_reason(self):
        paper = _make_paper()
        fr = FilterResult(paper=paper, relevant=False, reason="Not about driving")
        assert fr.relevant is False
        assert fr.reason == "Not about driving"

    def test_stores_paper_reference(self):
        paper = _make_paper(title="Specific Title")
        fr = FilterResult(paper=paper, relevant=True)
        assert fr.paper.title == "Specific Title"


class TestDigest:
    def test_rejected_field_default(self):
        d = Digest(date=datetime(2024, 1, 1), topic_name="Test")
        assert d.rejected == []
        assert isinstance(d.rejected, list)

    def test_rejected_field_with_entries(self):
        paper = _make_paper()
        rejected = [FilterResult(paper=paper, relevant=False, reason="Off topic")]
        d = Digest(
            date=datetime(2024, 1, 1),
            topic_name="Test",
            rejected=rejected,
        )
        assert len(d.rejected) == 1
        assert d.rejected[0].relevant is False
        assert d.rejected[0].reason == "Off topic"

    def test_existing_fields_preserved(self):
        """Ensure existing Digest fields still work."""
        entry = DigestEntry(
            paper=_make_paper(),
            scores=Scores(quality=0.9, llm_rank=1),
            rank=1,
            summary=Summary(one_liner="Test summary"),
        )
        d = Digest(
            date=datetime(2024, 1, 1),
            topic_name="Driving",
            entries=[entry],
            total_collected=10,
            total_new=5,
        )
        assert d.topic_name == "Driving"
        assert len(d.entries) == 1
        assert d.total_collected == 10
        assert d.total_new == 5
