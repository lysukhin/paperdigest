"""arXiv API collector."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone

import arxiv

from ..config import Config
from ..models import Paper
from .base import BaseCollector

logger = logging.getLogger(__name__)


class ArxivCollector(BaseCollector):
    """Collect papers from arXiv using keyword + category search."""

    source_name = "arXiv"

    def __init__(self, config: Config):
        super().__init__(config)
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3,
        )

    def collect(self) -> list[Paper]:
        topic = self.config.topic
        lookback = self.config.collection.lookback_days
        max_results = self.config.collection.max_results

        queries = self._build_queries(topic.primary_keywords, topic.arxiv_categories)
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback)

        all_papers: dict[str, Paper] = {}
        total_fetched = 0

        for qi, query in enumerate(queries, 1):
            # Extract keyword for readable log
            kw_match = re.search(r'"([^"]+)"', query)
            kw_label = kw_match.group(1) if kw_match else query[:40]
            logger.info(f"arXiv query {qi}/{len(queries)}: \"{kw_label}\"")

            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            query_fetched = 0
            query_kept = 0
            query_dupes = 0

            try:
                for result in self.client.results(search):
                    query_fetched += 1
                    pub_date = result.published
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                    if pub_date < cutoff:
                        continue

                    aid = result.entry_id.split("/")[-1]
                    aid = re.sub(r'v\d+$', '', aid)

                    if aid in all_papers:
                        query_dupes += 1
                        continue

                    updated = result.updated
                    if updated and updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)

                    paper = Paper(
                        arxiv_id=aid,
                        title=result.title.replace("\n", " ").strip(),
                        abstract=result.summary.replace("\n", " ").strip(),
                        authors=[a.name for a in result.authors],
                        published=pub_date,
                        updated=updated,
                        doi=result.doi,
                        categories=list(result.categories),
                        pdf_url=result.pdf_url,
                    )
                    all_papers[aid] = paper
                    query_kept += 1
            except Exception:
                logger.exception(f"Error fetching arXiv query: {query[:60]}")
                time.sleep(5)

            total_fetched += query_fetched
            logger.info(
                f"  → {query_fetched} fetched, {query_kept} new, "
                f"{query_dupes} cross-query dupes"
            )

        papers = list(all_papers.values())
        logger.info(
            f"Collected {len(papers)} unique papers from arXiv "
            f"({total_fetched} total fetched)"
        )
        return papers

    def _build_queries(
        self, keywords: list[str], categories: list[str]
    ) -> list[str]:
        """Build arXiv API queries from keywords and categories.

        Uses AND-joined individual words rather than exact phrases,
        since arXiv's API is more effective with boolean queries.
        """
        queries = []
        cat_filter = " OR ".join(f"cat:{c}" for c in categories) if categories else ""

        for kw in keywords:
            parts = kw.strip().split()
            if len(parts) == 1:
                term = f"all:{parts[0]}"
            elif len(parts) <= 3:
                # Short phrase: search as exact phrase in abstract
                term = f'abs:"{kw}"'
            else:
                # Longer phrase: AND-join key terms in abstract
                and_terms = " AND ".join(f"abs:{w}" for w in parts)
                term = f"({and_terms})"

            if cat_filter:
                queries.append(f"({term}) AND ({cat_filter})")
            else:
                queries.append(term)

        return queries
