"""NVIDIA Developer Blog collector via Atom feed."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from time import mktime

import feedparser

from ..config import Config
from ..models import Paper
from .base import BaseCollector

logger = logging.getLogger(__name__)

FEED_URL = "https://developer.nvidia.com/blog/feed"

# Keywords to filter AV/robotics-relevant posts
AV_KEYWORDS = [
    "autonomous driving", "self-driving", "autonomous vehicle",
    "lidar", "perception", "motion planning", "navigation",
    "robotics", "robot", "embodied",
    "vision language", "vlm", "vla",
    "point cloud", "3d detection", "object detection",
    "scene understanding", "isaac sim", "nvidia drive", "orin",
]


def _is_av_relevant(title: str, summary: str, tags: list[str]) -> bool:
    text = f"{title} {summary} {' '.join(tags)}".lower()
    return any(kw in text for kw in AV_KEYWORDS)


def _extract_slug(url: str) -> str:
    """Extract a slug from a blog URL."""
    # https://developer.nvidia.com/blog/some-post-title/ -> some-post-title
    parts = url.rstrip("/").split("/")
    return parts[-1] if parts else url


class NvidiaCollector(BaseCollector):
    """Collect AV-relevant posts from the NVIDIA Developer Blog Atom feed."""

    source_name = "NVIDIA Blog"

    def collect(self) -> list[Paper]:
        lookback = self.config.collection.lookback_days
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback)

        try:
            feed = feedparser.parse(FEED_URL)
        except Exception:
            logger.exception("Failed to fetch NVIDIA feed")
            return []

        if feed.bozo and not feed.entries:
            logger.warning(f"NVIDIA feed parse error: {feed.bozo_exception}")
            return []

        papers = []
        for entry in feed.entries:
            # Parse date
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if not published:
                continue
            pub_dt = datetime.fromtimestamp(mktime(published), tz=timezone.utc)
            if pub_dt < cutoff:
                continue

            title = entry.get("title", "").strip()
            # feedparser may include HTML in title
            title = re.sub(r"<[^>]+>", "", title).strip()
            summary = entry.get("summary", "").strip()
            summary = re.sub(r"<[^>]+>", "", summary).strip()

            if not title:
                continue

            tags = [t.term for t in entry.get("tags", [])]
            if not _is_av_relevant(title, summary, tags):
                continue

            link = entry.get("link", "")
            slug = _extract_slug(link)
            author = entry.get("author", "NVIDIA")

            # Check if post links to an arXiv paper
            arxiv_match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", summary)
            if arxiv_match:
                paper_id = arxiv_match.group(1)
            else:
                paper_id = f"nvidia:{slug}"

            papers.append(Paper(
                arxiv_id=paper_id,
                title=title,
                abstract=summary[:2000] if summary else title,
                authors=[author],
                published=pub_dt,
                pdf_url=link,
                categories=["blog:nvidia"],
            ))

        logger.info(f"Collected {len(papers)} posts from NVIDIA Blog")
        return papers
