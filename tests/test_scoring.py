"""Tests for the scoring module."""

from datetime import datetime, timezone

from paperdigest.config import (
    Config,
    CollectionConfig,
    QualityWeights,
    RelevanceWeights,
    ScoringConfig,
    TopicConfig,
)
from paperdigest.models import Paper, Scores
from paperdigest.scoring import compute_quality, compute_relevance, score_papers


def _make_config(**overrides) -> Config:
    topic = TopicConfig(
        name="VLM for AD",
        primary_keywords=["vision language model autonomous driving", "VLM autonomous driving"],
        secondary_keywords=["end-to-end driving", "scene understanding"],
        benchmarks=["nuScenes", "CARLA"],
        arxiv_categories=["cs.CV"],
    )
    scoring = ScoringConfig(
        alpha=0.65,
        relevance=RelevanceWeights(),
        quality=QualityWeights(),
        venue_tiers={
            "tier1": ["CVPR", "NeurIPS", "ICLR"],
            "tier2": ["ICRA", "IROS"],
            "tier3": ["ITSC"],
        },
    )
    return Config(topic=topic, scoring=scoring, **overrides)


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
        published=published or datetime.now(timezone.utc),
        citations=citations,
        max_hindex=max_hindex,
        venue=venue,
        code_url=code_url,
        code_official=code_official,
    )


class TestRelevanceScoring:
    def test_primary_keyword_hit(self):
        config = _make_config()
        paper = _make_paper(
            title="VLM autonomous driving system",
            abstract="We propose a new approach.",
        )
        score = compute_relevance(paper, config.scoring, config.topic)
        assert score >= 0.5  # primary_base

    def test_no_keyword_hit(self):
        config = _make_config()
        paper = _make_paper(
            title="Protein folding with transformers",
            abstract="Biology stuff",
        )
        score = compute_relevance(paper, config.scoring, config.topic)
        assert score == 0.0

    def test_secondary_keywords_add(self):
        config = _make_config()
        paper = _make_paper(
            title="VLM autonomous driving",
            abstract="We do end-to-end driving with scene understanding",
        )
        score = compute_relevance(paper, config.scoring, config.topic)
        # primary (0.5) + 2 secondary (0.2)
        assert score >= 0.7

    def test_benchmark_mentions(self):
        config = _make_config()
        paper = _make_paper(
            title="VLM autonomous driving",
            abstract="Evaluated on nuScenes and CARLA benchmarks",
        )
        score = compute_relevance(paper, config.scoring, config.topic)
        # primary (0.5) + 2 benchmarks (0.2)
        assert score >= 0.7

    def test_max_capped_at_one(self):
        config = _make_config()
        paper = _make_paper(
            title="VLM autonomous driving end-to-end driving scene understanding",
            abstract="nuScenes CARLA evaluation",
        )
        score = compute_relevance(paper, config.scoring, config.topic)
        assert score <= 1.0


class TestQualityScoring:
    def test_high_quality_paper(self):
        config = _make_config()
        paper = _make_paper(
            citations=100,
            max_hindex=60,
            venue="CVPR 2024",
            code_url="https://github.com/example/repo",
        )
        score = compute_quality(paper, config.scoring)
        assert score > 0.7

    def test_low_quality_paper(self):
        config = _make_config()
        paper = _make_paper(
            citations=0,
            max_hindex=2,
            venue=None,
            code_url=None,
        )
        score = compute_quality(paper, config.scoring)
        assert score < 0.5

    def test_freshness_decay(self):
        config = _make_config()
        from datetime import timedelta

        fresh = _make_paper(published=datetime.now(timezone.utc))
        old = _make_paper(
            arxiv_id="2401.00002",
            published=datetime.now(timezone.utc) - timedelta(days=25),
        )

        fresh_score = compute_quality(fresh, config.scoring)
        old_score = compute_quality(old, config.scoring)
        assert fresh_score > old_score


class TestRanking:
    def test_relevant_paper_ranks_higher(self):
        config = _make_config()
        relevant = _make_paper(
            arxiv_id="2401.00001",
            title="VLM autonomous driving with nuScenes",
            abstract="We propose end-to-end driving using VLM",
            citations=50,
            max_hindex=40,
            venue="CVPR",
            code_url="https://github.com/a/b",
        )
        irrelevant = _make_paper(
            arxiv_id="2401.00002",
            title="Protein structure prediction",
            abstract="Biology method for protein folding",
            citations=200,
            max_hindex=80,
            venue="Nature",
        )
        scored = score_papers([relevant, irrelevant], config)
        assert scored[0][0].arxiv_id == "2401.00001"

    def test_higher_quality_breaks_tie(self):
        config = _make_config()
        p1 = _make_paper(
            arxiv_id="2401.00001",
            title="VLM autonomous driving",
            abstract="A method",
            citations=100,
            venue="CVPR",
            code_url="https://github.com/a/b",
        )
        p2 = _make_paper(
            arxiv_id="2401.00002",
            title="VLM autonomous driving",
            abstract="Another method",
            citations=0,
            venue=None,
        )
        scored = score_papers([p1, p2], config)
        assert scored[0][0].arxiv_id == "2401.00001"
