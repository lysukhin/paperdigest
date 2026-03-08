"""Tests for the LLM paper scoring module."""

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
    """Tests for filter_paper() method (now scores instead of binary filter)."""

    def test_high_score_paper(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"score": 0.85, "reason": "Directly about VLMs for driving."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        result = f.filter_paper(_make_paper())

        assert isinstance(result, FilterResult)
        assert result.relevant is True
        assert result.score == 0.85
        assert result.reason == "Directly about VLMs for driving."

    def test_low_score_paper(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"score": 0.1, "reason": "About protein folding, not driving."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        paper = _make_paper(
            title="Protein Folding with Deep Learning",
            abstract="We study protein structure prediction.",
        )
        result = f.filter_paper(paper)

        assert isinstance(result, FilterResult)
        assert result.relevant is True  # no rejection, just low score
        assert result.score == 0.1
        assert result.reason == "About protein folding, not driving."

    def test_invalid_json_fails_open(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()
        f._client.chat.completions.create.return_value = _make_llm_response("not valid json {{{")

        result = f.filter_paper(_make_paper())

        assert isinstance(result, FilterResult)
        assert result.relevant is True
        assert result.score == 0.5
        assert "JSON" in result.reason or "parse" in result.reason.lower() or "fail" in result.reason.lower()

    def test_api_error_fails_open(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()
        f._client.chat.completions.create.side_effect = RuntimeError("API down")

        result = f.filter_paper(_make_paper())

        assert isinstance(result, FilterResult)
        assert result.relevant is True
        assert result.score == 0.5
        assert "error" in result.reason.lower() or "fail" in result.reason.lower()

    def test_tracks_token_usage(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"score": 0.7, "reason": "Relevant paper."})
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
        assert result.score == 0.5

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
        assert result.score == 0.5

    def test_uses_topic_description_in_system_prompt(self, db):
        config = _make_config(description="Papers about underwater robotics.")
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"score": 0.7, "reason": "Relevant."})
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

        resp_json = json.dumps({"score": 0.7, "reason": "Relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        paper = _make_paper(title="My Special Title", abstract="My special abstract content.")
        f.filter_paper(paper)

        call_args = f._client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = messages[1]["content"]
        assert "My Special Title" in user_msg
        assert "My special abstract content." in user_msg

    def test_score_clamped_to_valid_range(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"score": 1.5, "reason": "Over max."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)
        result = f.filter_paper(_make_paper())
        assert result.score == 1.0

    def test_negative_score_clamped(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"score": -0.2, "reason": "Under min."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)
        result = f.filter_paper(_make_paper())
        assert result.score == 0.0


class TestFilterPapers:
    """Tests for filter_papers() batch method."""

    def test_returns_all_papers_with_scores(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        f._client.chat.completions.create.side_effect = [
            _make_llm_response(json.dumps({"score": 0.9, "reason": "Great paper."})),
            _make_llm_response(json.dumps({"score": 0.1, "reason": "Off topic."})),
            _make_llm_response(json.dumps({"score": 0.7, "reason": "Good paper."})),
        ]

        papers = [
            _make_paper("2401.00001", "VLM for Driving"),
            _make_paper("2401.00002", "Protein Folding"),
            _make_paper("2401.00003", "Camera Calibration for AD"),
        ]
        for p in papers:
            p.db_id = db.upsert_paper(p)

        scored, rejected, score_map = f.filter_papers(papers)

        # All papers returned (no rejection)
        assert len(scored) == 3
        assert len(rejected) == 0
        # Score map populated
        assert score_map["2401.00001"] == pytest.approx(0.9)
        assert score_map["2401.00002"] == pytest.approx(0.1)
        assert score_map["2401.00003"] == pytest.approx(0.7)

    def test_stores_results_in_db(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"score": 0.8, "reason": "Relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        paper = _make_paper()
        paper.db_id = db.upsert_paper(paper)

        f.filter_papers([paper])

        results = db.get_filter_results()
        assert len(results) == 1
        assert results[0]["relevant"] == 1
        assert results[0]["reason"] == "Relevant."
        assert results[0]["score"] == pytest.approx(0.8)

    def test_writes_scores_to_scores_table(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"score": 0.75, "reason": "Good paper."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        paper = _make_paper()
        paper.db_id = db.upsert_paper(paper)

        f.filter_papers([paper])

        top = db.get_top_scored_papers(limit=1)
        assert len(top) == 1
        assert top[0][1].quality == pytest.approx(0.75)

    def test_logs_llm_usage_to_db(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"score": 0.7, "reason": "Relevant."})
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

        resp_json = json.dumps({"score": 0.7, "reason": "Relevant."})
        f._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        paper = _make_paper()
        paper.db_id = db.upsert_paper(paper)

        f.filter_papers([paper])

        assert f.run_id.startswith("filter_")

    def test_no_logging_when_zero_papers_scored(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        _, _, score_map = f.filter_papers([])

        stats = db.get_llm_stats()
        assert stats["runs"] == 0
        assert score_map == {}


class TestFilterProgress:
    """Tests for progress parameter in filter_papers()."""

    def test_progress_advance_called(self, db):
        config = _make_config()
        f = PaperFilter(config, db)
        f._client = MagicMock()

        resp_json = json.dumps({"score": 0.7, "reason": "Relevant."})
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

        resp_json = json.dumps({"score": 0.7, "reason": "Relevant."})
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

        # Fails open: paper gets neutral score
        assert result.relevant is True
        assert result.score == 0.5
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

        resp_json = json.dumps({"score": 0.3, "reason": "Low relevance."})
        f._client.chat.completions.create.return_value = _make_llm_response(
            resp_json, input_tokens=5000, output_tokens=3000
        )

        # First call succeeds (budget not yet consumed)
        r1 = f.filter_paper(_make_paper("2401.00001"))

        # Second call should fail open due to per-run budget
        r2 = f.filter_paper(_make_paper("2401.00002"))
        assert r2.relevant is True
        assert r2.score == 0.5
        assert "budget" in r2.reason.lower()


class TestFilterCaching:
    """Tests for filter result caching — skip papers already scored."""

    def test_skips_already_scored_paper(self, tmp_path):
        """Papers with a cached score are skipped without LLM call."""
        config = _make_config()
        db = Database(tmp_path / "test.db")
        db.init_schema()
        paper = _make_paper()
        paper_id = db.upsert_paper(paper)
        paper.db_id = paper_id

        # Pre-populate filter result with score
        db.upsert_filter_result(paper_id, relevant=True, reason="Good match", score=0.8)

        filt = PaperFilter(config, db)
        scored, rejected, score_map = filt.filter_papers([paper])

        assert len(scored) == 1
        assert len(rejected) == 0
        assert score_map[paper.arxiv_id] == pytest.approx(0.8)
        # No LLM call was made
        assert filt.run_papers == 0
        db.close()

    def test_old_cache_without_score_is_refetched(self, tmp_path):
        """Old cache entries without a score trigger a new LLM call."""
        config = _make_config()
        db = Database(tmp_path / "test.db")
        db.init_schema()
        paper = _make_paper()
        paper_id = db.upsert_paper(paper)
        paper.db_id = paper_id

        # Pre-populate old-style filter result (no score)
        db.upsert_filter_result(paper_id, relevant=True, reason="Good match", score=None)

        filt = PaperFilter(config, db)
        filt._client = MagicMock()
        resp_json = json.dumps({"score": 0.7, "reason": "Re-scored."})
        filt._client.chat.completions.create.return_value = _make_llm_response(resp_json)

        scored, rejected, score_map = filt.filter_papers([paper])

        assert len(scored) == 1
        assert score_map[paper.arxiv_id] == pytest.approx(0.7)
        # LLM was called because old cache had no score
        assert filt.run_papers == 1
        db.close()
