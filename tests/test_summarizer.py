"""Tests for the LLM summarizer module."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from paperdigest.config import (
    Config,
    CostControl,
    LLMConfig,
    SummarizerLLMConfig,
    TopicConfig,
)
from paperdigest.db import Database
from paperdigest.models import Paper, Summary
from paperdigest.summarizer import Summarizer


def _make_paper(arxiv_id="2401.00001", title="A Test Paper", **kwargs) -> Paper:
    defaults = dict(
        arxiv_id=arxiv_id,
        title=title,
        abstract="We propose a novel method for autonomous driving using vision language models.",
        authors=["Author One", "Author Two"],
        published=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return Paper(**defaults)


def _make_config(**overrides) -> Config:
    defaults = dict(
        topic=TopicConfig(name="AD", primary_keywords=["driving"], description="Papers about autonomous driving."),
        llm=LLMConfig(
            summarizer=SummarizerLLMConfig(
                enabled=True,
                model="gpt-5-nano",
                max_text_chars=50000,
                cost_control=CostControl(
                    max_cost_per_run=0.50,
                    max_cost_per_month=10.00,
                    input_cost_per_1k=0.00005,
                    output_cost_per_1k=0.0004,
                ),
            ),
        ),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_llm_response(content: str, input_tokens=100, output_tokens=200):
    """Build a mock OpenAI chat completion response."""
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


VALID_SUMMARY_JSON = json.dumps({
    "one_liner": "A novel VLM approach for autonomous driving.",
    "affiliations": "MIT, Waymo Research",
    "method": "We use a vision-language model to process driving scenes.",
    "data_benchmarks": "nuScenes, CARLA",
    "key_results": "95% accuracy on nuScenes planning benchmark.",
    "novelty": "First to apply VLMs directly to end-to-end driving.",
    "ad_relevance": "Directly targets autonomous driving perception.",
    "limitations": "Tested only in simulation environments.",
})


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.init_schema()
    yield d
    d.close()


class TestSummarizeOnePaper:
    """Tests for summarize_paper()."""

    def test_successful_summary(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.return_value = _make_llm_response(VALID_SUMMARY_JSON)

        result = s.summarize_paper(_make_paper())

        assert isinstance(result, Summary)
        assert result.one_liner == "A novel VLM approach for autonomous driving."
        assert result.affiliations == "MIT, Waymo Research"
        assert result.method == "We use a vision-language model to process driving scenes."
        assert result.data_benchmarks == "nuScenes, CARLA"
        assert result.key_results == "95% accuracy on nuScenes planning benchmark."
        assert result.novelty == "First to apply VLMs directly to end-to-end driving."
        assert result.ad_relevance == "Directly targets autonomous driving perception."
        assert result.limitations == "Tested only in simulation environments."

    def test_strips_markdown_fences(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()

        fenced = f"```json\n{VALID_SUMMARY_JSON}\n```"
        s._client.chat.completions.create.return_value = _make_llm_response(fenced)

        result = s.summarize_paper(_make_paper())
        assert result is not None
        assert result.one_liner == "A novel VLM approach for autonomous driving."

    def test_strips_bare_fences(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()

        fenced = f"```\n{VALID_SUMMARY_JSON}\n```"
        s._client.chat.completions.create.return_value = _make_llm_response(fenced)

        result = s.summarize_paper(_make_paper())
        assert result is not None

    def test_invalid_json_returns_none(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.return_value = _make_llm_response("not valid json {{{")

        result = s.summarize_paper(_make_paper())
        assert result is None

    def test_empty_response_returns_none(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()

        response = MagicMock()
        response.choices = []
        response.usage = None
        s._client.chat.completions.create.return_value = response

        result = s.summarize_paper(_make_paper())
        assert result is None

    def test_null_content_returns_none(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()

        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        response.usage = MagicMock(prompt_tokens=10, completion_tokens=10)
        s._client.chat.completions.create.return_value = response

        result = s.summarize_paper(_make_paper())
        assert result is None

    def test_api_exception_returns_none(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.side_effect = RuntimeError("API down")

        result = s.summarize_paper(_make_paper())
        assert result is None

    def test_missing_fields_default_to_empty(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()

        partial = json.dumps({"one_liner": "Partial response"})
        s._client.chat.completions.create.return_value = _make_llm_response(partial)

        result = s.summarize_paper(_make_paper())
        assert result is not None
        assert result.one_liner == "Partial response"
        assert result.method == ""
        assert result.limitations == ""

    def test_tracks_token_usage(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.return_value = _make_llm_response(
            VALID_SUMMARY_JSON, input_tokens=500, output_tokens=300
        )

        s.summarize_paper(_make_paper())

        assert s.run_input_tokens == 500
        assert s.run_output_tokens == 300
        assert s.run_papers == 1
        assert s.run_cost > 0


class TestBudgetEnforcement:
    """Tests for cost control / budget checking."""

    def test_per_run_budget_stops_summarization(self, db):
        config = _make_config(
            llm=LLMConfig(
                summarizer=SummarizerLLMConfig(
                    enabled=True,
                    cost_control=CostControl(max_cost_per_run=0.001),
                ),
            ),
        )
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.return_value = _make_llm_response(
            VALID_SUMMARY_JSON, input_tokens=5000, output_tokens=3000
        )

        # First call succeeds (budget not yet consumed)
        r1 = s.summarize_paper(_make_paper("2401.00001"))
        assert r1 is not None

        # Second call should be skipped -- run cost exceeds budget
        r2 = s.summarize_paper(_make_paper("2401.00002"))
        assert r2 is None

    def test_monthly_budget_exhausted(self, db):
        config = _make_config(
            llm=LLMConfig(
                summarizer=SummarizerLLMConfig(
                    enabled=True,
                    cost_control=CostControl(max_cost_per_month=0.01),
                ),
            ),
        )

        # Simulate prior usage that exhausted the monthly budget
        db.log_llm_usage(
            run_id="old-run",
            papers_summarized=100,
            input_tokens=100000,
            output_tokens=50000,
            estimated_cost=0.05,
        )

        s = Summarizer(config, db)
        s._client = MagicMock()

        result = s.summarize_paper(_make_paper())
        assert result is None
        # Should not have called the API at all
        s._client.chat.completions.create.assert_not_called()


class TestSummarizeMultiple:
    """Tests for summarize_papers() batch method."""

    def test_summarizes_multiple_papers(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.return_value = _make_llm_response(VALID_SUMMARY_JSON)

        papers = [_make_paper(f"2401.0000{i}") for i in range(3)]
        results = s.summarize_papers(papers)

        assert len(results) == 3
        for arxiv_id, summary in results.items():
            assert isinstance(summary, Summary)
            assert summary.one_liner != ""

    def test_logs_usage_to_db(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.return_value = _make_llm_response(
            VALID_SUMMARY_JSON, input_tokens=100, output_tokens=200
        )

        s.summarize_papers([_make_paper()])

        stats = db.get_llm_stats()
        assert stats["runs"] == 1
        assert stats["total_papers_summarized"] == 1
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 200
        assert stats["total_cost_usd"] > 0

    def test_skips_failures_continues_rest(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()

        # First call fails, second succeeds
        s._client.chat.completions.create.side_effect = [
            RuntimeError("API error"),
            _make_llm_response(VALID_SUMMARY_JSON),
        ]

        papers = [_make_paper("2401.00001"), _make_paper("2401.00002")]
        results = s.summarize_papers(papers)

        assert len(results) == 1
        assert "2401.00002" in results

    def test_no_logging_when_zero_papers_summarized(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.side_effect = RuntimeError("all fail")

        s.summarize_papers([_make_paper()])

        stats = db.get_llm_stats()
        assert stats["runs"] == 0


class TestSummarizeProgress:
    """Tests for progress parameter in summarize_papers()."""

    def test_progress_advance_called(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.return_value = _make_llm_response(VALID_SUMMARY_JSON)

        papers = [_make_paper(f"2401.0000{i}") for i in range(3)]

        progress = MagicMock()
        s.summarize_papers(papers, progress=progress)

        assert progress.advance.call_count == 3
        assert progress.set_cost.call_count == 3

    def test_no_progress_uses_logger(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.return_value = _make_llm_response(VALID_SUMMARY_JSON)

        # Default progress=None should use logger (no error)
        s.summarize_papers([_make_paper()])


class TestCostEstimation:
    """Tests for _estimate_cost()."""

    def test_default_rates(self, db):
        config = _make_config()
        s = Summarizer(config, db)

        cost = s._estimate_cost(1000, 1000)
        cc = config.llm.summarizer.cost_control
        expected = (1000 / 1000) * cc.input_cost_per_1k + (1000 / 1000) * cc.output_cost_per_1k
        assert cost == pytest.approx(expected)

    def test_zero_tokens_zero_cost(self, db):
        config = _make_config()
        s = Summarizer(config, db)

        assert s._estimate_cost(0, 0) == 0.0

    def test_custom_rates(self, db):
        config = _make_config(
            llm=LLMConfig(
                summarizer=SummarizerLLMConfig(
                    enabled=True,
                    cost_control=CostControl(input_cost_per_1k=0.01, output_cost_per_1k=0.03),
                ),
            ),
        )
        s = Summarizer(config, db)

        cost = s._estimate_cost(2000, 1000)
        assert cost == pytest.approx(0.01 * 2 + 0.03 * 1)


class TestClientInitialization:
    """Tests for lazy client creation."""

    def test_missing_api_key_raises(self, db, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        config = _make_config()
        s = Summarizer(config, db)

        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            with pytest.raises(RuntimeError, match="LLM_API_KEY"):
                _ = s.client

    def test_missing_openai_package_raises(self, db):
        config = _make_config()
        s = Summarizer(config, db)

        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(RuntimeError, match="openai package not installed"):
                _ = s.client


class TestBuildMessages:
    """Tests for _build_messages() — always tries full text."""

    def test_uses_full_text_when_pdf_available(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        paper = _make_paper(pdf_url="https://arxiv.org/pdf/2401.00001.pdf")

        with patch("paperdigest.pdf.fetch_paper_text", return_value="Full paper text here..."):
            messages = s._build_messages(paper)

        assert "Full paper text here..." in messages[1]["content"]
        assert "full text" in messages[0]["content"].lower()

    def test_falls_back_to_abstract_when_pdf_extraction_fails(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        paper = _make_paper(pdf_url="https://arxiv.org/pdf/2401.00001.pdf")

        with patch("paperdigest.pdf.fetch_paper_text", return_value=None):
            messages = s._build_messages(paper)

        assert "Abstract" in messages[1]["content"]
        assert paper.abstract in messages[1]["content"]

    def test_falls_back_to_abstract_when_no_pdf_url(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        paper = _make_paper()  # no pdf_url or oa_pdf_url

        messages = s._build_messages(paper)

        assert "Abstract" in messages[1]["content"]
        assert paper.abstract in messages[1]["content"]

    def test_prefers_oa_pdf_url(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        paper = _make_paper(
            pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
            oa_pdf_url="https://openaccess.example.com/paper.pdf",
        )

        with patch("paperdigest.pdf.fetch_paper_text", return_value="OA text") as mock_fetch:
            s._build_messages(paper)

        mock_fetch.assert_called_once_with(
            "https://openaccess.example.com/paper.pdf",
            max_chars=config.llm.summarizer.max_text_chars,
        )


class TestSummarizerExtraInstructions:
    def test_extra_instructions_appended(self, db):
        """Extra instructions appended to summarizer system prompt."""
        config = _make_config()
        config.llm.summarizer.extra_instructions = "Include benchmark results in one_liner"
        summarizer = Summarizer(config, db)
        paper = Paper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract about driving.",
            authors=["Author"],
            published=datetime.now(timezone.utc),
        )
        messages = summarizer._build_messages(paper)
        assert "Additional instructions:" in messages[0]["content"]
        assert "Include benchmark results in one_liner" in messages[0]["content"]

    def test_no_extra_instructions(self, db):
        """Without extra_instructions, prompt is unchanged."""
        config = _make_config()
        summarizer = Summarizer(config, db)
        paper = Paper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract.",
            authors=["Author"],
            published=datetime.now(timezone.utc),
        )
        messages = summarizer._build_messages(paper)
        assert "Additional instructions:" not in messages[0]["content"]


class TestSummaryCaching:
    """Tests for DB-backed summary caching."""

    def test_returns_cached_summary_without_llm_call(self, db):
        config = _make_config()
        paper = _make_paper(db_id=None)
        # Insert paper into DB to get db_id
        paper.db_id = db.upsert_paper(paper)

        # Pre-cache a summary
        cached = Summary(one_liner="Cached result", method="Cached method")
        db.upsert_summary(paper.db_id, cached)

        s = Summarizer(config, db)
        s._client = MagicMock()

        result = s.summarize_paper(paper)

        assert result is not None
        assert result.one_liner == "Cached result"
        assert result.method == "Cached method"
        # LLM should NOT have been called
        s._client.chat.completions.create.assert_not_called()

    def test_caches_new_summary_to_db(self, db):
        config = _make_config()
        paper = _make_paper(db_id=None)
        paper.db_id = db.upsert_paper(paper)

        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.return_value = _make_llm_response(VALID_SUMMARY_JSON)

        result = s.summarize_paper(paper)
        assert result is not None

        # Verify it was persisted to DB
        cached = db.get_summary(paper.db_id)
        assert cached is not None
        assert cached.one_liner == result.one_liner

    def test_no_caching_when_paper_has_no_db_id(self, db):
        """Papers without db_id skip caching (shouldn't happen in practice)."""
        config = _make_config()
        paper = _make_paper()  # db_id=None by default
        assert paper.db_id is None

        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.return_value = _make_llm_response(VALID_SUMMARY_JSON)

        result = s.summarize_paper(paper)
        assert result is not None
        # LLM was called since no caching possible
        s._client.chat.completions.create.assert_called_once()


class TestRanking:
    """Tests for rank_papers() LLM ranking."""

    def test_rank_papers_returns_correct_ranking(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()

        papers = [
            _make_paper("2401.00001", "Paper A"),
            _make_paper("2401.00002", "Paper B"),
            _make_paper("2401.00003", "Paper C"),
        ]
        summaries = {
            "2401.00001": Summary(one_liner="First paper"),
            "2401.00002": Summary(one_liner="Second paper"),
            "2401.00003": Summary(one_liner="Third paper"),
        }
        quality_scores = {
            "2401.00001": 0.8,
            "2401.00002": 0.6,
            "2401.00003": 0.9,
        }

        ranking_json = json.dumps({"ranking": ["2401.00003", "2401.00001", "2401.00002"]})
        s._client.chat.completions.create.return_value = _make_llm_response(ranking_json)

        result = s.rank_papers(papers, summaries, quality_scores)

        assert result == {"2401.00003": 1, "2401.00001": 2, "2401.00002": 3}

    def test_fallback_ranking_on_llm_failure(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()
        s._client.chat.completions.create.side_effect = RuntimeError("API down")

        papers = [
            _make_paper("2401.00001", "Paper A"),
            _make_paper("2401.00002", "Paper B"),
            _make_paper("2401.00003", "Paper C"),
        ]
        summaries = {
            "2401.00001": Summary(one_liner="First paper"),
            "2401.00002": Summary(one_liner="Second paper"),
            "2401.00003": Summary(one_liner="Third paper"),
        }
        quality_scores = {
            "2401.00001": 0.5,
            "2401.00002": 0.9,
            "2401.00003": 0.7,
        }

        result = s.rank_papers(papers, summaries, quality_scores)

        # Fallback orders by quality score descending
        assert result == {"2401.00002": 1, "2401.00003": 2, "2401.00001": 3}

    def test_missing_papers_in_llm_response_appended(self, db):
        config = _make_config()
        s = Summarizer(config, db)
        s._client = MagicMock()

        papers = [
            _make_paper("2401.00001", "Paper A"),
            _make_paper("2401.00002", "Paper B"),
            _make_paper("2401.00003", "Paper C"),
        ]
        summaries = {
            "2401.00001": Summary(one_liner="First paper"),
            "2401.00002": Summary(one_liner="Second paper"),
            "2401.00003": Summary(one_liner="Third paper"),
        }
        quality_scores = {
            "2401.00001": 0.8,
            "2401.00002": 0.6,
            "2401.00003": 0.9,
        }

        # LLM only returns 1 paper in ranking
        ranking_json = json.dumps({"ranking": ["2401.00002"]})
        s._client.chat.completions.create.return_value = _make_llm_response(ranking_json)

        result = s.rank_papers(papers, summaries, quality_scores)

        # 2401.00002 is rank 1 from LLM, remaining appended by quality score desc
        assert result["2401.00002"] == 1
        # The other two should be appended — 2401.00003 has higher quality (0.9) than 2401.00001 (0.8)
        assert result["2401.00003"] == 2
        assert result["2401.00001"] == 3
