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


def enrich_paper(paper: Paper, config: Config) -> Paper:
    """Enrich a single paper with Semantic Scholar data."""
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
            resp = requests.get(
                url, headers=headers, params={"fields": FIELDS}, timeout=10
            )
            if resp.status_code == 429:
                logger.warning("S2 rate limited, sleeping 60s...")
                time.sleep(60)
                resp = requests.get(
                    url, headers=headers, params={"fields": FIELDS}, timeout=10
                )

            if resp.status_code == 404:
                logger.debug(f"S2 not found: {ident}")
                continue

            resp.raise_for_status()
            data = resp.json()

            paper.citations = data.get("citationCount")
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


def enrich_papers(papers: list[Paper], config: Config) -> list[Paper]:
    """Enrich a list of papers. Respects rate limits."""
    api_key = config.semantic_scholar_api_key
    delay = 0.5 if api_key else 3.5  # Unauthenticated: ~100 req/5min

    enriched = []
    for i, paper in enumerate(papers):
        logger.info(f"Enriching [{i+1}/{len(papers)}] {paper.arxiv_id}")
        paper = enrich_paper(paper, config)
        enriched.append(paper)
        if i < len(papers) - 1:
            time.sleep(delay)

    return enriched
