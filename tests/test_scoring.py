"""Tests for the scoring module (quality-only, relevance handled by LLM filter)."""

from datetime import datetime, timedelta, timezone

import pytest

from paperdigest.config import (
    Config,
    QualityWeights,
    ScoringConfig,
    TopicConfig,
)
from paperdigest.models import Paper, Scores
from paperdigest.scoring import compute_quality, score_papers

# Fixed reference time for deterministic tests — set to "now" at import time
# so all tests within a single run share the same value
REF_TIME = datetime.now(timezone.utc)


def _make_config(**overrides):
    scoring_data = {
        "quality": QualityWeights(
            w_venue=0.35, w_code=0.30, w_fresh=0.35
        ),
        "venue_tiers": {
            "tier1": ["CVPR", "NeurIPS", "ICML", "ICLR"],
            "tier2": ["IROS", "ICRA"],
            "tier3": ["IV", "ITSC"],
        },
    }
    scoring_data.update(overrides.pop("scoring", {}))
    return Config(
        topic=TopicConfig(name="Test", primary_keywords=["test"]),
        scoring=ScoringConfig(**scoring_data),
        **overrides,
    )


def _make_paper(
    title="A Paper",
    abstract="An abstract",
    citations=None,
    max_hindex=None,
    venue=None,
    code_url=None,
    code_official=False,
    published=None,
    arxiv_id="2401.00001",
) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=["Alice", "Bob"],
        published=published or REF_TIME,
        citations=citations,
        max_hindex=max_hindex,
        venue=venue,
        code_url=code_url,
        code_official=code_official,
    )


class TestQualityScoring:
    def test_high_quality_paper(self):
        config = _make_config()
        paper = _make_paper(
            venue="CVPR 2024",
            code_url="https://github.com/example/repo",
            published=REF_TIME,
        )
        score = compute_quality(paper, config.scoring)
        assert score > 0.7

    def test_low_quality_paper(self):
        config = _make_config()
        paper = _make_paper(
            venue=None,
            code_url=None,
            published=REF_TIME - timedelta(days=60),
        )
        score = compute_quality(paper, config.scoring)
        assert score < 0.5

    def test_freshness_decay(self):
        config = _make_config()

        fresh = _make_paper(published=REF_TIME)
        old = _make_paper(
            arxiv_id="2401.00002",
            published=REF_TIME - timedelta(days=25),
        )

        fresh_score = compute_quality(fresh, config.scoring)
        old_score = compute_quality(old, config.scoring)
        assert fresh_score > old_score


class TestScorePapers:
    def test_returns_papers_with_quality_scores(self):
        config = _make_config()
        papers = [
            _make_paper(
                arxiv_id="2401.00001",
                venue="CVPR",
                code_url="https://github.com/a/b",
            ),
            _make_paper(
                arxiv_id="2401.00002",
                venue=None,
            ),
        ]
        results = score_papers(papers, config)

        assert len(results) == 2
        for paper, scores in results:
            assert isinstance(paper, Paper)
            assert isinstance(scores, Scores)
            assert scores.quality >= 0
            assert scores.llm_rank == 0

        # Results should be sorted by quality descending
        assert results[0][1].quality >= results[1][1].quality
