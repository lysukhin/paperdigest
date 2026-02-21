"""Data models for the paper digest system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    """Raw paper collected from a source."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    updated: datetime | None = None
    doi: str | None = None
    categories: list[str] = field(default_factory=list)
    pdf_url: str | None = None
    # Enrichment fields
    citations: int | None = None
    max_hindex: int | None = None
    venue: str | None = None
    oa_pdf_url: str | None = None
    code_url: str | None = None
    code_official: bool = False
    # Internal
    db_id: int | None = None


@dataclass
class Scores:
    """Computed scores for a paper."""

    relevance: float = 0.0
    quality: float = 0.0
    final: float = 0.0


@dataclass
class Summary:
    """LLM-generated structured summary."""

    one_liner: str = ""
    affiliations: str = ""
    method: str = ""
    data_benchmarks: str = ""
    key_results: str = ""
    novelty: str = ""
    ad_relevance: str = ""
    limitations: str = ""


@dataclass
class DigestEntry:
    """A single entry in a digest."""

    paper: Paper
    scores: Scores
    rank: int
    summary: Summary | None = None


@dataclass
class Digest:
    """A complete digest ready for delivery."""

    date: datetime
    topic_name: str
    entries: list[DigestEntry] = field(default_factory=list)
    total_collected: int = 0
    total_new: int = 0
