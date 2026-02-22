"""Semantic Scholar API enrichment."""

from __future__ import annotations

import logging
import time

import requests

from ..config import Config
from ..models import Paper

logger = logging.getLogger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "citationCount,venue,openAccessPdf,authors.hIndex"

MAX_RETRIES = 3
INITIAL_BACKOFF = 60


def enrich_paper(paper: Paper, config: Config) -> Paper:
    """Enrich a single paper with Semantic Scholar data. Mutates paper in-place."""
    api_key = config.semantic_scholar_api_key
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    # Try arXiv ID first, then DOI
    identifiers = [f"ARXIV:{paper.arxiv_id}"]
    if paper.doi:
        identifiers.append(f"DOI:{paper.doi}")

    for ident in identifiers:
        url = f"{S2_API_BASE}/paper/{ident}"
        try:
            resp = _request_with_backoff(url, headers, retries=MAX_RETRIES)
            if resp is None:
                continue

            if resp.status_code == 404:
                logger.debug(f"S2 not found: {ident}")
                continue

            resp.raise_for_status()
            data = resp.json()

            # Use 0 instead of None so get_unenriched_papers doesn't re-process
            paper.citations = data.get("citationCount") or 0
            paper.venue = data.get("venue") or paper.venue

            # Max h-index from authors
            authors = data.get("authors", [])
            h_indices = [a.get("hIndex") for a in authors if a.get("hIndex") is not None]
            if h_indices:
                paper.max_hindex = max(h_indices)

            # Open access PDF
            oa = data.get("openAccessPdf")
            if oa and oa.get("url"):
                paper.oa_pdf_url = oa["url"]

            logger.debug(f"Enriched from S2: {paper.arxiv_id} (cites={paper.citations})")
            return paper

        except requests.RequestException:
            logger.debug(f"S2 request failed for {ident}")
            continue

    return paper


def _request_with_backoff(
    url: str, headers: dict, retries: int = MAX_RETRIES
) -> requests.Response | None:
    """Make a GET request with exponential backoff on 429 responses."""
    backoff = INITIAL_BACKOFF
    for attempt in range(retries):
        resp = requests.get(
            url, headers=headers, params={"fields": FIELDS}, timeout=10
        )
        if resp.status_code != 429:
            return resp
        if attempt < retries - 1:
            logger.warning(f"S2 rate limited, sleeping {backoff}s (attempt {attempt + 1}/{retries})...")
            time.sleep(backoff)
            backoff *= 2
        else:
            logger.warning("S2 rate limit: max retries exhausted, skipping")
            return None
    return None


def enrich_papers(papers: list[Paper], config: Config, progress=None) -> list[Paper]:
    """Enrich a list of papers. Respects rate limits."""
    api_key = config.semantic_scholar_api_key
    delay = 0.5 if api_key else 3.5  # Unauthenticated: ~100 req/5min

    consecutive_429s = 0
    max_consecutive_429s = 5

    enriched = []
    for i, paper in enumerate(papers):
        if progress is not None:
            progress.advance(1)
        else:
            logger.info(f"Enriching [{i+1}/{len(papers)}] {paper.arxiv_id}")

        # Circuit breaker: stop if too many consecutive rate limits
        if consecutive_429s >= max_consecutive_429s:
            logger.warning("Circuit breaker: too many consecutive 429s, stopping S2 enrichment")
            enriched.append(paper)
            continue

        paper = enrich_paper(paper, config)
        enriched.append(paper)

        # Track consecutive 429s by checking if enrichment succeeded
        if paper.citations is None:
            consecutive_429s += 1
        else:
            consecutive_429s = 0

        if i < len(papers) - 1:
            time.sleep(delay)

    return enriched
