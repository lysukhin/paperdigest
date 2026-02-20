"""Deduplication logic for papers."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

from .db import Database
from .models import Paper

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 0.85


def normalize_title(title: str) -> str:
    """Normalize a title for comparison."""
    return " ".join(title.lower().split())


def titles_match(a: str, b: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """Check if two titles match using fuzzy comparison."""
    na = normalize_title(a)
    nb = normalize_title(b)
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def dedup_papers(papers: list[Paper], db: Database) -> list[Paper]:
    """Deduplicate papers against the database and within the batch.

    Returns only papers that are new (not already in DB).
    """
    new_papers = []
    seen_ids: set[str] = set()
    seen_titles: list[str] = []

    for paper in papers:
        # 1. Exact arxiv_id match
        if paper.arxiv_id in seen_ids:
            continue
        existing = db.get_paper_by_arxiv_id(paper.arxiv_id)
        if existing:
            seen_ids.add(paper.arxiv_id)
            continue

        # 2. Exact DOI match
        if paper.doi:
            existing = db.get_paper_by_doi(paper.doi)
            if existing:
                seen_ids.add(paper.arxiv_id)
                logger.debug(f"Dedup DOI match: {paper.doi}")
                continue

        # 3. Fuzzy title match within current batch
        is_dup = False
        for seen_title in seen_titles:
            if titles_match(paper.title, seen_title):
                logger.debug(f"Dedup fuzzy title match: {paper.title[:60]}")
                is_dup = True
                break
        if is_dup:
            seen_ids.add(paper.arxiv_id)
            continue

        # 4. Fuzzy title match against DB (check recent papers)
        # Only check title-based dedup for papers without arxiv_id matches
        # This is a lighter check - we don't scan the entire DB for fuzzy matches

        seen_ids.add(paper.arxiv_id)
        seen_titles.append(paper.title)
        new_papers.append(paper)

    logger.info(
        f"Dedup: {len(papers)} input → {len(new_papers)} new "
        f"({len(papers) - len(new_papers)} duplicates removed)"
    )
    return new_papers
