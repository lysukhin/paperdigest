"""Paper scoring: quality signals only (relevance handled by LLM filter)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from .config import Config, QualityWeights, ScoringConfig
from .models import Paper, Scores

logger = logging.getLogger(__name__)


def _venue_tier_score(venue: str | None, venue_tiers: dict[str, list[str]]) -> float:
    """Map venue to tier score using word-boundary matching."""
    if not venue:
        return 0.2
    for tier, tier_score in [("tier1", 1.0), ("tier2", 0.7), ("tier3", 0.4)]:
        for v in venue_tiers.get(tier, []):
            if re.search(r'\b' + re.escape(v) + r'\b', venue, re.IGNORECASE):
                return tier_score
    return 0.2


def compute_quality(paper: Paper, config: ScoringConfig) -> float:
    """Compute quality score (0-1) as weighted sum of signals."""
    qw: QualityWeights = config.quality

    venue_score = _venue_tier_score(paper.venue, config.venue_tiers)
    code_score = 1.0 if paper.code_url else 0.0

    published = (
        paper.published.astimezone(timezone.utc)
        if paper.published.tzinfo
        else paper.published.replace(tzinfo=timezone.utc)
    )
    age_days = (datetime.now(timezone.utc) - published).days
    fresh_score = max(0.0, 1.0 - age_days / 30.0)

    score = (
        qw.w_venue * venue_score
        + qw.w_code * code_score
        + qw.w_fresh * fresh_score
    )

    return min(score, 1.0)


def score_papers(papers: list[Paper], config: Config) -> list[tuple[Paper, Scores]]:
    """Compute quality scores for all papers. Returns list of (Paper, Scores).

    Note: llm_rank is set to 0 here — ranking is done by the summarizer.
    """
    results = []
    for paper in papers:
        quality = compute_quality(paper, config.scoring)
        scores = Scores(quality=quality, llm_rank=0)
        results.append((paper, scores))

    results.sort(key=lambda x: x[1].quality, reverse=True)
    logger.info(f"Scored {len(results)} papers (quality only)")
    if results:
        top = results[0]
        logger.info(f"Top quality: [{top[1].quality:.3f}] {top[0].title[:70]}")
    return results
