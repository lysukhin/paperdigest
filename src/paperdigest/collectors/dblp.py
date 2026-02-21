"""DBLP conference paper collector via search API."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

import requests

from ..config import Config
from ..models import Paper
from .base import BaseCollector

logger = logging.getLogger(__name__)

SEARCH_URL = "https://dblp.org/search/publ/api"

# Search keywords scoped to AD/VLM — combined with venue+year
SEARCH_KEYWORDS = [
    "autonomous+driving",
    "self-driving",
    "motion+planning+driving",
    "end-to-end+driving",
]

# Map config venue names to DBLP search terms
# Some short acronyms cause 500 errors — use longer forms where needed
VENUE_SEARCH_TERMS: dict[str, str] = {
    "CVPR": "CVPR",
    "ICCV": "ICCV",
    "ECCV": "ECCV",
    "NeurIPS": "Neural Information Processing",  # "NeurIPS" triggers DBLP 500
    "ICML": "ICML",
    "ICLR": "ICLR",
    "AAAI": "AAAI",
    "CoRL": "CoRL",
    "RSS": "Robotics Science Systems",  # "RSS" alone is ambiguous
    "ICRA": "ICRA",
    "IROS": "IROS",
    "WACV": "WACV",
    "IV": "IV",
    "ITSC": "ITSC",
}


def _clean_author_name(name: str) -> str:
    """Remove DBLP disambiguation suffixes like '0001'."""
    return re.sub(r"\s+\d{4}$", "", name).strip()


def _extract_arxiv_id(ee: str) -> str | None:
    """Try to extract an arXiv ID from a DBLP electronic edition URL."""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", ee)
    return m.group(1) if m else None


class DBLPCollector(BaseCollector):
    """Collect AD-relevant papers from DBLP conference proceedings."""

    source_name = "DBLP Conferences"

    def collect(self) -> list[Paper]:
        conf_cfg = self.config.collection.conferences
        if not conf_cfg.enabled:
            return []

        venues = conf_cfg.venues
        years = _target_years(conf_cfg.years_back)

        papers_by_key: dict[str, Paper] = {}

        for venue in venues:
            search_term = VENUE_SEARCH_TERMS.get(venue, venue)
            for year in years:
                for keyword in SEARCH_KEYWORDS:
                    new = self._query(keyword, search_term, year)
                    for p in new:
                        if p.arxiv_id not in papers_by_key:
                            papers_by_key[p.arxiv_id] = p
                    # Polite rate limit — DBLP asks for 1 req/sec
                    time.sleep(1.0)

        papers = list(papers_by_key.values())
        logger.info(f"Collected {len(papers)} papers from DBLP ({len(venues)} venues)")
        return papers

    def _query(self, keyword: str, venue_term: str, year: int) -> list[Paper]:
        """Run a single DBLP search query."""
        q = f"{keyword} {venue_term} {year}"
        try:
            resp = requests.get(
                SEARCH_URL,
                params={"q": q, "format": "json", "h": 100},
                timeout=15,
            )
            if resp.status_code != 200:
                logger.debug(f"DBLP query failed ({resp.status_code}): {q}")
                return []
            data = resp.json()
        except Exception:
            logger.debug(f"DBLP request error: {q}", exc_info=True)
            return []

        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        papers = []

        for hit in hits:
            info = hit.get("info", {})
            title = info.get("title", "").rstrip(".")
            if not title:
                continue

            # Only accept conference papers
            pub_type = info.get("type", "")
            if "Conference" not in pub_type and "Workshop" not in pub_type:
                continue

            # Parse authors
            raw_authors = info.get("authors", {}).get("author", [])
            if isinstance(raw_authors, dict):
                raw_authors = [raw_authors]
            authors = [
                _clean_author_name(a["text"] if isinstance(a, dict) else a)
                for a in raw_authors
            ]

            venue = info.get("venue", "")
            pub_year = int(info.get("year", year))
            doi = info.get("doi", "")
            ee = info.get("ee", "")
            dblp_key = info.get("key", "")

            # Resolve paper ID — prefer arXiv if available
            arxiv_id = _extract_arxiv_id(ee) if ee else None
            if not arxiv_id:
                paper_id = f"dblp:{dblp_key}" if dblp_key else f"dblp:{_slugify(title)}"
            else:
                paper_id = arxiv_id

            published = datetime(pub_year, 1, 1, tzinfo=timezone.utc)

            papers.append(Paper(
                arxiv_id=paper_id,
                title=title,
                abstract="",  # DBLP doesn't provide abstracts
                authors=authors,
                published=published,
                pdf_url=ee or "",
                categories=[f"conf:{venue}"],
                doi=doi or None,
            ))

        return papers


def _target_years(years_back: int) -> list[int]:
    """Return the list of years to query."""
    current = datetime.now(timezone.utc).year
    return list(range(current, current - years_back - 1, -1))


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return slug.strip("-")[:80]
