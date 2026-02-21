"""Waymo Research publications collector via HTML scraping."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

from ..config import Config
from ..models import Paper
from .base import BaseCollector

logger = logging.getLogger(__name__)

RESEARCH_URL = "https://waymo.com/research/"


class WaymoCollector(BaseCollector):
    """Collect papers from Waymo's research publications page."""

    source_name = "Waymo Research"

    def collect(self) -> list[Paper]:
        lookback = self.config.collection.lookback_days

        try:
            resp = requests.get(RESEARCH_URL, timeout=30)
            resp.raise_for_status()
        except Exception:
            logger.exception("Failed to fetch Waymo research page")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        papers = []

        # Find all publication entries — Waymo uses structured HTML
        # with paper titles as links and metadata in surrounding elements
        for link_el in soup.find_all("a", href=True):
            href = link_el["href"]

            # Look for arXiv links
            arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", href)
            if not arxiv_match:
                continue

            arxiv_id = arxiv_match.group(1)

            # Walk up to find the paper's container and title
            title = _extract_title(link_el)
            if not title:
                continue

            authors = _extract_authors(link_el)

            # Waymo's research page doesn't have per-paper dates, so we use
            # a year filter from the page's filtering metadata if available
            year = _extract_year(link_el)

            # For lookback filtering: if we can extract a year, skip old papers
            # Otherwise include all (the page is curated, not huge)
            if year:
                try:
                    pub_dt = datetime(year, 6, 1, tzinfo=timezone.utc)  # approximate
                    cutoff_year = datetime.now(timezone.utc).year
                    if year < cutoff_year - 1:
                        continue
                except ValueError:
                    pass

            papers.append(Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract="",  # Will be filled by arXiv/S2 enrichment
                authors=authors if authors else ["Waymo Research"],
                published=datetime(year or datetime.now().year, 1, 1, tzinfo=timezone.utc),
                pdf_url=href,
                categories=["research:waymo"],
            ))

        # Deduplicate by arxiv_id within this batch
        seen = {}
        for p in papers:
            if p.arxiv_id not in seen:
                seen[p.arxiv_id] = p

        papers = list(seen.values())
        logger.info(f"Collected {len(papers)} papers from Waymo Research")
        return papers


def _extract_title(link_el) -> str:
    """Try to find the paper title near a link element."""
    # Check the link text itself
    text = link_el.get_text(strip=True)
    if text and len(text) > 10 and "download" not in text.lower():
        return text

    # Walk up and look for a heading or strong text
    for parent in link_el.parents:
        if parent.name in ("article", "div", "li", "section"):
            heading = parent.find(["h2", "h3", "h4", "strong"])
            if heading:
                return heading.get_text(strip=True)
            break
    return ""


def _extract_authors(link_el) -> list[str]:
    """Try to extract author names near a link element."""
    for parent in link_el.parents:
        if parent.name in ("article", "div", "li", "section"):
            # Look for a list of author names (often in <ul> or comma-separated text)
            for el in parent.find_all(["ul", "p", "span"]):
                text = el.get_text(strip=True)
                if "," in text and len(text) < 500:
                    # Looks like an author list
                    names = [n.strip() for n in text.split(",") if n.strip()]
                    if 2 <= len(names) <= 30:
                        return names
            break
    return []


def _extract_year(link_el) -> int | None:
    """Try to extract publication year from nearby metadata."""
    for parent in link_el.parents:
        if parent.name in ("article", "div", "li", "section"):
            text = parent.get_text()
            match = re.search(r"\b(20[12]\d)\b", text)
            if match:
                return int(match.group(1))
            break
    return None
