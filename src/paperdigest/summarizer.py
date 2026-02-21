"""LLM-based paper summarization with cost controls."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime

from .config import Config
from .db import Database
from .models import Paper, Summary

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_ABSTRACT = """You are a research paper analyst specializing in autonomous driving and computer vision.
Given a paper's title and abstract, produce a structured JSON summary with these fields:
- one_liner: A single sentence capturing the core contribution (max 150 chars)
- affiliations: Comma-separated list of author affiliations/institutions (e.g. "MIT, Google Research, Tsinghua University")
- method: Brief description of the proposed method/approach (2-3 sentences)
- data_benchmarks: Datasets and benchmarks used for evaluation
- key_results: Most important quantitative or qualitative results
- novelty: What makes this work novel compared to prior work
- ad_relevance: How this work relates to autonomous driving applications
- limitations: Key limitations or open questions (be honest but brief)

Respond with ONLY valid JSON, no markdown fences."""

SYSTEM_PROMPT_FULL_TEXT = """You are a research paper analyst specializing in autonomous driving and computer vision.
Given a paper's title and full text, produce a structured JSON summary with these fields:
- one_liner: A single sentence capturing the core contribution (max 150 chars)
- affiliations: Comma-separated list of author affiliations/institutions (e.g. "MIT, Google Research, Tsinghua University")
- method: Detailed description of the proposed method/approach (3-5 sentences)
- data_benchmarks: Datasets and benchmarks used for evaluation, with specific metrics reported
- key_results: Most important quantitative results with numbers
- novelty: What makes this work novel compared to prior work
- ad_relevance: How this work relates to autonomous driving applications
- limitations: Key limitations or open questions (be honest but brief)

Respond with ONLY valid JSON, no markdown fences."""

USER_TEMPLATE_ABSTRACT = """Paper title: {title}

Abstract: {abstract}"""

USER_TEMPLATE_FULL_TEXT = """Paper title: {title}

Full text:
{full_text}"""


class Summarizer:
    """LLM summarizer with cost tracking and budget enforcement."""

    def __init__(self, config: Config, db: Database):
        self.config = config
        self.db = db
        self.run_id = str(uuid.uuid4())[:8]
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
                base_url=self.config.llm.base_url,
            )
        return self._client

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        cc = self.config.llm.cost_control
        return (input_tokens / 1000) * cc.input_cost_per_1k + (
            output_tokens / 1000
        ) * cc.output_cost_per_1k

    def _check_budget(self) -> tuple[bool, str]:
        """Check if we're within budget. Returns (ok, reason)."""
        cc = self.config.llm.cost_control

        # Check monthly budget
        monthly_cost = self.db.get_monthly_llm_cost()
        if monthly_cost >= cc.max_cost_per_month:
            return False, f"Monthly budget exhausted (${monthly_cost:.2f} / ${cc.max_cost_per_month:.2f})"

        warn_threshold = cc.max_cost_per_month * cc.warn_at_percent / 100
        if monthly_cost >= warn_threshold:
            logger.warning(
                f"LLM budget warning: ${monthly_cost:.2f} / ${cc.max_cost_per_month:.2f} "
                f"({monthly_cost / cc.max_cost_per_month * 100:.0f}%)"
            )

        # Check per-run budget using adaptive estimate
        if self.run_papers > 0:
            avg_cost = self.run_cost / self.run_papers
            estimated_next = avg_cost * 1.5  # safety margin
        else:
            estimated_next = self._estimate_cost(800, 500)  # conservative first estimate
        if self.run_cost + estimated_next > cc.max_cost_per_run:
            return False, f"Per-run budget would be exceeded (${self.run_cost:.4f} / ${cc.max_cost_per_run:.2f})"

        return True, ""

    def _build_messages(self, paper: Paper) -> list[dict]:
        """Build the LLM messages, using full text if enabled and available."""
        use_full = self.config.llm.use_full_text
        full_text = None

        if use_full:
            pdf_url = paper.oa_pdf_url or paper.pdf_url
            if pdf_url:
                from .pdf import fetch_paper_text

                logger.info(f"Fetching full text for {paper.arxiv_id}...")
                full_text = fetch_paper_text(
                    pdf_url, max_chars=self.config.llm.max_text_chars
                )
                if full_text:
                    logger.info(
                        f"Extracted {len(full_text)} chars from PDF"
                    )
                else:
                    logger.warning(
                        f"Full text extraction failed for {paper.arxiv_id}, "
                        "falling back to abstract"
                    )
            else:
                logger.warning(
                    f"No PDF URL for {paper.arxiv_id}, falling back to abstract"
                )

        if full_text:
            system = SYSTEM_PROMPT_FULL_TEXT
            user = USER_TEMPLATE_FULL_TEXT.format(
                title=paper.title, full_text=full_text
            )
        else:
            system = SYSTEM_PROMPT_ABSTRACT
            user = USER_TEMPLATE_ABSTRACT.format(
                title=paper.title, abstract=paper.abstract
            )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def summarize_paper(self, paper: Paper) -> Summary | None:
        """Summarize a single paper. Returns None on failure."""
        ok, reason = self._check_budget()
        if not ok:
            logger.warning(f"Skipping summary for {paper.arxiv_id}: {reason}")
            return None

        messages = self._build_messages(paper)

        try:
            kwargs = dict(
                model=self.config.llm.model,
                messages=messages,
            )
            if self.config.llm.temperature is not None:
                kwargs["temperature"] = self.config.llm.temperature
            if self.config.llm.max_completion_tokens is not None:
                kwargs["max_completion_tokens"] = self.config.llm.max_completion_tokens
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
                logger.warning(f"Empty LLM response for {paper.arxiv_id}")
                return None

            # Parse response
            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present (handles ```json, ``` with trailing whitespace, etc.)
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```\s*$', '', content)
            content = content.strip()

            data = json.loads(content)
            self.run_papers += 1

            return Summary(
                one_liner=data.get("one_liner", ""),
                affiliations=data.get("affiliations", ""),
                method=data.get("method", ""),
                data_benchmarks=data.get("data_benchmarks", ""),
                key_results=data.get("key_results", ""),
                novelty=data.get("novelty", ""),
                ad_relevance=data.get("ad_relevance", ""),
                limitations=data.get("limitations", ""),
            )

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM JSON for {paper.arxiv_id}")
            return None
        except Exception:
            logger.exception(f"LLM call failed for {paper.arxiv_id}")
            return None

    def summarize_papers(self, papers: list[Paper]) -> dict[str, Summary]:
        """Summarize multiple papers. Returns {arxiv_id: Summary}."""
        summaries: dict[str, Summary] = {}

        for i, paper in enumerate(papers):
            logger.info(f"Summarizing [{i+1}/{len(papers)}] {paper.arxiv_id}")
            summary = self.summarize_paper(paper)
            if summary:
                summaries[paper.arxiv_id] = summary

        # Log usage for this run
        if self.run_papers > 0:
            self.db.log_llm_usage(
                run_id=self.run_id,
                papers_summarized=self.run_papers,
                input_tokens=self.run_input_tokens,
                output_tokens=self.run_output_tokens,
                estimated_cost=self.run_cost,
            )
            logger.info(
                f"LLM run complete: {self.run_papers} papers, "
                f"${self.run_cost:.4f} estimated cost"
            )

        return summaries
