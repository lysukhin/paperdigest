"""Tests for the LLM summarizer module."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from paperdigest.config import Config, CostControl, LLMConfig, TopicConfig
from paperdigest.db import Database
from paperdigest.models import Paper, Summary
from paperdigest.summarizer import Summarizer


def _make_paper(arxiv_id="2401.00001", title="A Test Paper") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="We propose a novel method for autonomous driving using vision language models.",
        authors=["Author One", "Author Two"],
        published=datetime.now(timezone.utc),
    )


def _make_config(**overrides) -> Config:
    defaults = dict(
        topic=TopicConfig(name="Test", primary_keywords=["test"]),
        llm=LLMConfig(
            enabled=True,
            model="gpt-4o-mini",
            cost_control=CostControl(
                max_cost_per_run=0.50,
                max_cost_per_month=10.00,
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
                enabled=True,
                cost_control=CostControl(max_cost_per_run=0.001),
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

        # Second call should be skipped — run cost exceeds budget
        r2 = s.summarize_paper(_make_paper("2401.00002"))
        assert r2 is None

    def test_monthly_budget_exhausted(self, db):
        config = _make_config(
            llm=LLMConfig(
                enabled=True,
                cost_control=CostControl(max_cost_per_month=0.01),
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


class TestCostEstimation:
    """Tests for _estimate_cost()."""

    def test_default_rates(self, db):
        config = _make_config()
        s = Summarizer(config, db)

        cost = s._estimate_cost(1000, 1000)
        cc = config.llm.cost_control
        expected = (1000 / 1000) * cc.input_cost_per_1k + (1000 / 1000) * cc.output_cost_per_1k
        assert cost == pytest.approx(expected)

    def test_zero_tokens_zero_cost(self, db):
        config = _make_config()
        s = Summarizer(config, db)

        assert s._estimate_cost(0, 0) == 0.0

    def test_custom_rates(self, db):
        config = _make_config(
            llm=LLMConfig(
                enabled=True,
                cost_control=CostControl(input_cost_per_1k=0.01, output_cost_per_1k=0.03),
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
