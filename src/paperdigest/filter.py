"""LLM-based paper relevance filtering with cost controls."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime

from .config import Config
from .db import Database
from .models import FilterResult, Paper

logger = logging.getLogger(__name__)

FILTER_SYSTEM_PROMPT = """You are a research paper relevance filter.

The user is interested in:
{description}

Given a paper's title and abstract, decide if it is relevant to these interests.
Respond with ONLY valid JSON: {{"relevant": true, "reason": "one sentence explaining why"}}
or {{"relevant": false, "reason": "one sentence explaining why not"}}"""

FILTER_USER_TEMPLATE = """Paper title: {title}

Abstract: {abstract}"""


class PaperFilter:
    """LLM-based paper filter with cost tracking and budget enforcement.

    Fails open: on any error (JSON parse, API error, budget exhaustion),
    papers are considered relevant.
    """

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.run_id = f"filter_{uuid.uuid4().hex[:8]}"
        self.run_cost = 0.0
        self.run_input_tokens = 0
        self.run_output_tokens = 0
        self.run_papers = 0
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Install with: pip install paperdigest[llm]"
                )
            api_key = self.config.llm_api_key
            if not api_key:
                raise RuntimeError("LLM_API_KEY environment variable not set")
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config.llm.filter.base_url,
            )
        return self._client

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        cc = self.config.llm.filter.cost_control
        return (input_tokens / 1000) * cc.input_cost_per_1k + (
            output_tokens / 1000
        ) * cc.output_cost_per_1k

    def _check_budget(self) -> tuple[bool, str]:
        """Check if we're within budget. Returns (ok, reason)."""
        cc = self.config.llm.filter.cost_control

        # Check monthly budget
        monthly_cost = self.db.get_monthly_llm_cost()
        if monthly_cost >= cc.max_cost_per_month:
            return False, f"Monthly budget exhausted (${monthly_cost:.2f} / ${cc.max_cost_per_month:.2f})"

        warn_threshold = cc.max_cost_per_month * cc.warn_at_percent / 100
        if monthly_cost >= warn_threshold:
            logger.warning(
                f"Filter LLM budget warning: ${monthly_cost:.2f} / ${cc.max_cost_per_month:.2f} "
                f"({monthly_cost / cc.max_cost_per_month * 100:.0f}%)"
            )

        # Check per-run budget using adaptive estimate
        if self.run_papers > 0:
            avg_cost = self.run_cost / self.run_papers
            estimated_next = avg_cost * 1.5  # safety margin
        else:
            estimated_next = self._estimate_cost(300, 100)  # conservative first estimate
        if self.run_cost + estimated_next > cc.max_cost_per_run:
            return False, f"Per-run budget would be exceeded (${self.run_cost:.4f} / ${cc.max_cost_per_run:.2f})"

        return True, ""

    def _build_messages(self, paper: Paper) -> list[dict]:
        """Build the LLM messages for filtering."""
        system = FILTER_SYSTEM_PROMPT.format(
            description=self.config.topic.description,
        )
        extra = self.config.llm.filter.extra_instructions
        if extra:
            system += f"\n\nAdditional instructions:\n{extra}"
        user = FILTER_USER_TEMPLATE.format(
            title=paper.title,
            abstract=paper.abstract,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def filter_paper(self, paper: Paper) -> FilterResult:
        """Filter a single paper. Fails open on any error."""
        # Check budget -- fail open if exhausted
        ok, reason = self._check_budget()
        if not ok:
            logger.warning(f"Filter budget exceeded for {paper.arxiv_id}: {reason}")
            return FilterResult(paper=paper, relevant=True, reason=f"Filter skipped: {reason}")

        messages = self._build_messages(paper)

        try:
            kwargs = dict(
                model=self.config.llm.filter.model,
                messages=messages,
            )
            if self.config.llm.filter.temperature is not None:
                kwargs["temperature"] = self.config.llm.filter.temperature
            if self.config.llm.filter.max_completion_tokens is not None:
                kwargs["max_completion_tokens"] = self.config.llm.filter.max_completion_tokens
            response = self.client.chat.completions.create(**kwargs)

            # Track usage
            usage = response.usage
            if usage:
                input_t = usage.prompt_tokens
                output_t = usage.completion_tokens
                cost = self._estimate_cost(input_t, output_t)
                self.run_input_tokens += input_t
                self.run_output_tokens += output_t
                self.run_cost += cost

            # Guard against empty/null response
            if not response.choices or response.choices[0].message.content is None:
                logger.warning(f"Empty LLM response for filter on {paper.arxiv_id}")
                return FilterResult(paper=paper, relevant=True, reason="Filter failed: empty LLM response")

            # Parse response
            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```\s*$', '', content)
            content = content.strip()

            data = json.loads(content)
            self.run_papers += 1

            return FilterResult(
                paper=paper,
                relevant=bool(data.get("relevant", True)),
                reason=data.get("reason", ""),
            )

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse filter JSON for {paper.arxiv_id}")
            return FilterResult(paper=paper, relevant=True, reason="Filter failed: JSON parse error")
        except Exception:
            logger.exception(f"Filter LLM call failed for {paper.arxiv_id}")
            return FilterResult(paper=paper, relevant=True, reason="Filter failed: API error")

    def filter_papers(
        self, papers: list[Paper], progress=None
    ) -> tuple[list[Paper], list[FilterResult]]:
        """Filter multiple papers. Returns (relevant_papers, rejected_results)."""
        relevant_papers: list[Paper] = []
        rejected_results: list[FilterResult] = []

        for i, paper in enumerate(papers):
            if progress is not None:
                progress.advance(1)
                progress.set_cost(self.run_cost)
            else:
                logger.info(f"Filtering [{i+1}/{len(papers)}] {paper.arxiv_id}")

            # Check for cached filter result
            if paper.db_id is not None:
                cached = self.db.get_latest_filter_result(paper.db_id)
                if cached is not None:
                    if cached["relevant"]:
                        relevant_papers.append(paper)
                    else:
                        rejected_results.append(
                            FilterResult(paper=paper, relevant=False, reason=cached["reason"])
                        )
                    continue

            result = self.filter_paper(paper)

            # Store result in DB if paper has a db_id
            if paper.db_id is not None:
                self.db.upsert_filter_result(
                    paper_id=paper.db_id,
                    relevant=result.relevant,
                    reason=result.reason,
                )

            if result.relevant:
                relevant_papers.append(paper)
            else:
                rejected_results.append(result)

        # Log LLM usage for this run
        if self.run_papers > 0:
            self.db.log_llm_usage(
                run_id=self.run_id,
                papers_summarized=self.run_papers,
                input_tokens=self.run_input_tokens,
                output_tokens=self.run_output_tokens,
                estimated_cost=self.run_cost,
            )
            logger.info(
                f"Filter run complete: {len(relevant_papers)} relevant, "
                f"{len(rejected_results)} rejected out of {len(papers)} papers, "
                f"${self.run_cost:.4f} estimated cost"
            )

        return relevant_papers, rejected_results
