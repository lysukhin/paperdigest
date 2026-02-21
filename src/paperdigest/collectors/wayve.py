"""Wayve Science publications collector via HTML scraping."""

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

SCIENCE_URL = "https://wayve.ai/science/"

# Month name mapping for parsing dates like "26 Mar 2025"
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date(text: str) -> datetime | None:
    """Parse dates like '26 Mar 2025' or 'March 2025'."""
    text = text.strip()

    # Try "DD Mon YYYY"
    m = re.match(r"(\d{1,2})\s+(\w{3,})\s+(\d{4})", text)
    if m:
        day = int(m.group(1))
        month_str = m.group(2)[:3].lower()
        year = int(m.group(3))
        month = MONTHS.get(month_str)
        if month:
            return datetime(year, month, day, tzinfo=timezone.utc)

    # Try "Mon YYYY" or "Month YYYY"
    m = re.match(r"(\w{3,})\s+(\d{4})", text)
    if m:
        month_str = m.group(1)[:3].lower()
        year = int(m.group(2))
        month = MONTHS.get(month_str)
        if month:
            return datetime(year, month, 1, tzinfo=timezone.utc)

    return None


def _slugify(title: str) -> str:
    """Create a URL-safe slug from a title."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return slug.strip("-")[:80]


class WayveCollector(BaseCollector):
    """Collect papers from Wayve's science publications page."""

    source_name = "Wayve Science"

    def collect(self) -> list[Paper]:
        lookback = self.config.collection.lookback_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback)

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; paperdigest/0.1; research aggregator)"
        }
        try:
            resp = requests.get(SCIENCE_URL, headers=headers, timeout=30)
            resp.raise_for_status()
        except Exception:
            logger.exception("Failed to fetch Wayve science page")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        papers = []

        # Find all "Download paper" links which point to PDFs
        for link_el in soup.find_all("a", href=True):
            href = link_el["href"]
            link_text = link_el.get_text(strip=True).lower()

            # Look for PDF download links
            is_pdf = href.endswith(".pdf") or "download" in link_text
            if not is_pdf:
                continue

            # Find the paper's container
            title, pub_date = _extract_paper_info(link_el)
            if not title:
                continue

            # Filter by lookback period
            if pub_date and pub_date < cutoff:
                continue

            # Check if this links to arXiv
            arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", href)
            if arxiv_match:
                paper_id = arxiv_match.group(1)
            else:
                paper_id = f"wayve:{_slugify(title)}"

            papers.append(Paper(
                arxiv_id=paper_id,
                title=title,
                abstract="",  # Wayve doesn't provide abstracts on the page
                authors=["Wayve"],
                published=pub_date or datetime.now(timezone.utc),
                pdf_url=href if href.startswith("http") else f"https://wayve.ai{href}",
                categories=["research:wayve"],
            ))

        # Deduplicate within batch
        seen = {}
        for p in papers:
            if p.arxiv_id not in seen:
                seen[p.arxiv_id] = p

        papers = list(seen.values())
        logger.info(f"Collected {len(papers)} papers from Wayve Science")
        return papers


def _extract_paper_info(link_el) -> tuple[str, datetime | None]:
    """Extract title and date from the vicinity of a PDF link."""
    title = ""
    pub_date = None

    # Walk up to find the paper container
    for parent in link_el.parents:
        if parent.name in ("article", "div", "li", "section", "tr"):
            # Look for the title — usually a heading or strong text before the link
            heading = parent.find(["h2", "h3", "h4", "strong", "b"])
            if heading:
                title = heading.get_text(strip=True)
            else:
                # Try getting the largest text block that isn't the link itself
                for el in parent.find_all(["p", "span", "div"]):
                    text = el.get_text(strip=True)
                    if len(text) > len(title) and "download" not in text.lower():
                        title = text

            # Look for a date
            text = parent.get_text()
            date_match = re.search(
                r"(\d{1,2}\s+\w{3,}\s+\d{4}|\w{3,}\s+\d{4})", text
            )
            if date_match:
                pub_date = _parse_date(date_match.group(1))

            if title:
                break

    return title.strip(), pub_date
