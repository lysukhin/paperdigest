"""Tests for the LLM paper filter module."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from paperdigest.config import (
    Config,
    CostControl,
    FilterLLMConfig,
    LLMConfig,
    TopicConfig,
)
from paperdigest.db import Database
from paperdigest.filter import PaperFilter
from paperdigest.models import FilterResult, Paper


def _make_paper(
    arxiv_id="2401.00001",
    title="Vision Language Models for Autonomous Driving",
    abstract="We propose a novel VLM approach for self-driving cars.",
) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=["Author One", "Author Two"],
        published=datetime.now(timezone.utc),
    )


def _make_config(
    max_cost_per_run=0.10,
    max_cost_per_month=3.00,
    description="Papers about vision-language models for autonomous driving.",
    filter_enabled=True,
) -> Config:
    return Config(
        topic=TopicConfig(
            name="Test",
            primary_keywords=["test"],
            description=description,
        ),
        llm=LLMConfig(
            filter=FilterLLMConfig(
                enabled=filter_enabled,
                model="gpt-4o-mini",
                cost_control=CostControl(
                    max_cost_per_run=max_cost_per_run,
                    max_cost_per_month=max_cost_per_month,
                    input_cost_per_1k=0.00015,
                    output_cost_per_1k=0.0006,
                ),
            ),
        ),
    )


def _make_llm_response(content: str, input_tokens=50, output_tokens=30):
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


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.init_schema()
    yield d
    d.close()


class TestFilterPaper:
    """Tests for filter_paper() method."""

    def test_relevant_paper(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": True, "reason": "Directly about VLMs for driving."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        result = f.filter_paper(_make_paper())

        assert isinstance(result, FilterResult)
        assert result.relevant is True
        assert result.reason == "Directly about VLMs for driving."

    def test_irrelevant_paper(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": False, "reason": "About protein folding, not driving."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        paper = _make_paper(
            title="Protein Folding with Deep Learning",
            abstract="We study protein structure prediction.",
        )
        result = f.filter_paper(paper)

        assert isinstance(result, FilterResult)
        assert result.relevant is False
        assert result.reason == "About protein folding, not driving."

    def test_invalid_json_fails_open(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()
        f._client.chat.completions.create.return_value = _make_llm_response("not valid json {{{")

        result = f.filter_paper(_make_paper())

        assert isinstance(result, FilterResult)
        assert result.relevant is True
        assert "JSON" in result.reason or "parse" in result.reason.lower() or "fail" in result.reason.lower()

    def test_api_error_fails_open(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()
        f._client.chat.completions.create.side_effect = RuntimeError("API down")

        result = f.filter_paper(_make_paper())

        assert isinstance(result, FilterResult)
        assert result.relevant is True
        assert "error" in result.reason.lower() or "fail" in result.reason.lower()

    def test_tracks_token_usage(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": True, "reason": "Relevant paper."})
        f._client.chat.completions.create.return_value = _make_llm_response(
            resp_json, input_tokens=100, output_tokens=50
        )

        f.filter_paper(_make_paper())

        assert f.run_input_tokens == 100
        assert f.run_output_tokens == 50
        assert f.run_papers == 1
        assert f.run_cost > 0

    def test_empty_response_fails_open(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        response = MagicMock()
        response.choices = []
        response.usage = None
        f._client.chat.completions.create.return_value = response

        result = f.filter_paper(_make_paper())

        assert result.relevant is True

    def test_null_content_fails_open(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        response.usage = MagicMock(prompt_tokens=10, completion_tokens=10)
        f._client.chat.completions.create.return_value = response

        result = f.filter_paper(_make_paper())

        assert result.relevant is True

    def test_uses_topic_description_in_system_prompt(self, db):
        config = _make_config(description="Papers about underwater robotics.")
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": True, "reason": "Relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        f.filter_paper(_make_paper())

        call_args = f._client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = messages[0]["content"]
        assert "underwater robotics" in system_msg

    def test_sends_title_and_abstract_in_user_message(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": True, "reason": "Relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        paper = _make_paper(title="My Special Title", abstract="My special abstract content.")
        f.filter_paper(paper)

        call_args = f._client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = messages[1]["content"]
        assert "My Special Title" in user_msg
        assert "My special abstract content." in user_msg


class TestFilterPapers:
    """Tests for filter_papers() batch method."""

    def test_splits_relevant_and_rejected(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        relevant_json = json.dumps({"relevant": True, "reason": "Relevant to driving."})
        rejected_json = json.dumps({"relevant": False, "reason": "Not about driving."})

        f._client.chat.completions.create.side_effect = [
            _make_llm_response(relevant_json),
            _make_llm_response(rejected_json),
            _make_llm_response(relevant_json),
        ]

        papers = [
            _make_paper("2401.00001", "VLM for Driving"),
            _make_paper("2401.00002", "Protein Folding"),
            _make_paper("2401.00003", "Camera Calibration for AD"),
        ]
        # Papers need db_ids for upsert_filter_result
        for p in papers:
            p.db_id = db.upsert_paper(p)

        relevant, rejected = f.filter_papers(papers)

        assert len(relevant) == 2
        assert len(rejected) == 1
        assert relevant[0].arxiv_id == "2401.00001"
        assert relevant[1].arxiv_id == "2401.00003"
        assert rejected[0].paper.arxiv_id == "2401.00002"
        assert rejected[0].relevant is False

    def test_stores_results_in_db(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": True, "reason": "Relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        paper = _make_paper()
        paper.db_id = db.upsert_paper(paper)

        f.filter_papers([paper])

        results = db.get_filter_results()
        assert len(results) == 1
        assert results[0]["relevant"] == 1
        assert results[0]["reason"] == "Relevant."

    def test_logs_llm_usage_to_db(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": True, "reason": "Relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(
            resp_json, input_tokens=100, output_tokens=50
        )

        paper = _make_paper()
        paper.db_id = db.upsert_paper(paper)

        f.filter_papers([paper])

        stats = db.get_llm_stats()
        assert stats["runs"] == 1
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 50
        assert stats["total_cost_usd"] > 0

    def test_run_id_has_filter_prefix(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": True, "reason": "Relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        paper = _make_paper()
        paper.db_id = db.upsert_paper(paper)

        f.filter_papers([paper])

        assert f.run_id.startswith("filter_")

    def test_no_logging_when_zero_papers_filtered(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        f.filter_papers([])

        stats = db.get_llm_stats()
        assert stats["runs"] == 0


class TestFilterProgress:
    """Tests for progress parameter in filter_papers()."""

    def test_progress_advance_called(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": True, "reason": "Relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        papers = [
            _make_paper("2401.00001"),
            _make_paper("2401.00002"),
            _make_paper("2401.00003"),
        ]
        for p in papers:
            p.db_id = db.upsert_paper(p)

        progress = MagicMock()
        f.filter_papers(papers, progress=progress)

        assert progress.advance.call_count == 3
        assert progress.set_cost.call_count == 3

    def test_no_progress_uses_logger(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": True, "reason": "Relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        paper = _make_paper()
        paper.db_id = db.upsert_paper(paper)

        # Default progress=None should use logger (no error)
        f.filter_papers([paper])


class TestFilterExtraInstructions:
    def test_extra_instructions_appended(self, db):
        """Extra instructions are appended to filter system prompt."""
        config = _make_config()
        config.llm.filter.extra_instructions = "Focus on robotics applications"
        filt = PaperFilter(config, db)
        paper = Paper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author"],
            published=datetime.now(timezone.utc),
        )
        messages = filt._build_messages(paper)
        assert "Additional instructions:" in messages[0]["content"]
        assert "Focus on robotics applications" in messages[0]["content"]

    def test_no_extra_instructions(self, db):
        """Without extra_instructions, prompt is unchanged."""
        config = _make_config()
        filt = PaperFilter(config, db)
        paper = Paper(
            arxiv_id="2401.00001",
            title="Test Paper",
            abstract="Test abstract",
            authors=["Author"],
            published=datetime.now(timezone.utc),
        )
        messages = filt._build_messages(paper)
        assert "Additional instructions:" not in messages[0]["content"]


class TestFilterBudget:
    """Tests for budget enforcement in filter."""

    def test_budget_exhaustion_fails_open(self, db):
        config = _make_config(max_cost_per_month=0.001)

        # Simulate prior usage that exhausted the monthly budget
        db.log_llm_usage(
            run_id="old-run",
            papers_summarized=100,
            input_tokens=100000,
            output_tokens=50000,
            estimated_cost=0.05,
        )

        f = PaperFilter(config, db)
        f._client = MagicMock()

        result = f.filter_paper(_make_paper())

        # Fails open: paper is considered relevant
        assert result.relevant is True
        assert "budget" in result.reason.lower()

    def test_budget_exhaustion_api_not_called(self, db):
        config = _make_config(max_cost_per_month=0.001)

        db.log_llm_usage(
            run_id="old-run",
            papers_summarized=100,
            input_tokens=100000,
            output_tokens=50000,
            estimated_cost=0.05,
        )

        f = PaperFilter(config, db)
        f._client = MagicMock()

        f.filter_paper(_make_paper())

        # API should NOT have been called
        f._client.chat.completions.create.assert_not_called()

    def test_per_run_budget_fails_open(self, db):
        config = _make_config(max_cost_per_run=0.0001)
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"relevant": False, "reason": "Not relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(
            resp_json, input_tokens=5000, output_tokens=3000
        )

        # First call succeeds (budget not yet consumed)
        r1 = f.filter_paper(_make_paper("2401.00001"))
        # May be relevant=False from the LLM response

        # Second call should fail open due to per-run budget
        r2 = f.filter_paper(_make_paper("2401.00002"))
        assert r2.relevant is True
        assert "budget" in r2.reason.lower()
