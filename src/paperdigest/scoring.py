"""Paper scoring: relevance + quality -> final score."""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone

from .config import Config, QualityWeights, RelevanceWeights, ScoringConfig
from .models import Paper, Scores

logger = logging.getLogger(__name__)


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Case-insensitive keyword match."""
    return keyword.lower() in text.lower()


def compute_relevance(paper: Paper, config: ScoringConfig, topic_cfg) -> float:
    """Compute relevance score (0-1) based on keyword matching."""
    rw: RelevanceWeights = config.relevance
    text = f"{paper.title} {paper.abstract}"
    score = 0.0

    # Primary keyword hit
    has_primary = any(
        _keyword_in_text(kw, text) for kw in topic_cfg.primary_keywords
    )
    if has_primary:
        score += rw.primary_base

    # Secondary keywords
    secondary_score = 0.0
    for kw in topic_cfg.secondary_keywords:
        if _keyword_in_text(kw, text):
            secondary_score += rw.secondary_increment
    score += min(secondary_score, rw.secondary_cap)

    # Benchmark mentions
    bench_score = 0.0
    for bm in topic_cfg.benchmarks:
        if _keyword_in_text(bm, text):
            bench_score += rw.benchmark_increment
    score += min(bench_score, rw.benchmark_cap)

    return min(score, 1.0)


def _venue_tier_score(venue: str | None, venue_tiers: dict[str, list[str]]) -> float:
    """Map venue to tier score using word-boundary matching for short names."""
    if not venue:
        return 0.2

    for tier, tier_score in [("tier1", 1.0), ("tier2", 0.7), ("tier3", 0.4)]:
        for v in venue_tiers.get(tier, []):
            # Use word-boundary matching to avoid false positives
            # (e.g. "IV" matching "driving")
            if re.search(r'\b' + re.escape(v) + r'\b', venue, re.IGNORECASE):
                return tier_score
    return 0.2


def compute_quality(paper: Paper, config: ScoringConfig) -> float:
    """Compute quality score (0-1) as weighted sum of signals."""
    qw: QualityWeights = config.quality

    venue_score = _venue_tier_score(paper.venue, config.venue_tiers)
    author_score = min(1.0, (paper.max_hindex or 0) / 50.0)
    cite_score = min(1.0, math.log(1 + (paper.citations or 0)) / 5.0)
    code_score = 1.0 if paper.code_url else 0.0

    # Freshness: linearly decay over 30 days
    published = paper.published.astimezone(timezone.utc) if paper.published.tzinfo else paper.published.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - published).days
    fresh_score = max(0.0, 1.0 - age_days / 30.0)

    score = (
        qw.w_venue * venue_score
        + qw.w_author * author_score
        + qw.w_cite * cite_score
        + qw.w_code * code_score
        + qw.w_fresh * fresh_score
    )

    return min(score, 1.0)


def score_paper(paper: Paper, config: Config) -> Scores:
    """Compute full scores for a paper."""
    relevance = compute_relevance(paper, config.scoring, config.topic)
    quality = compute_quality(paper, config.scoring)
    alpha = config.scoring.alpha
    final = alpha * relevance + (1 - alpha) * quality
    return Scores(relevance=relevance, quality=quality, final=final)


def score_papers(papers: list[Paper], config: Config) -> list[tuple[Paper, Scores]]:
    """Score all papers and return sorted by final score descending."""
    results = []
    for paper in papers:
        scores = score_paper(paper, config)
        results.append((paper, scores))

    results.sort(key=lambda x: x[1].final, reverse=True)
    logger.info(f"Scored {len(results)} papers")
    if results:
        top = results[0]
        logger.info(
            f"Top paper: [{top[1].final:.3f}] {top[0].title[:70]}"
        )
    return results
