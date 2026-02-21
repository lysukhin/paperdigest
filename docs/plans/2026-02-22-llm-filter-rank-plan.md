# LLM-Driven Filtering and Ranking — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace keyword-based relevance scoring with a two-tier LLM architecture: cheap model filters papers by relevance, good model summarizes full text and ranks survivors.

**Architecture:** New pipeline: fetch → dedup → store → LLM filter (cheap model, binary relevant/not) → enrich (survivors only) → quality score → LLM summarize + rank (good model, full text always) → digest. Rejected papers tracked in DB and shown in digest footer.

**Tech Stack:** Python 3.11+, OpenAI API (two models), SQLite, pytest, Jinja2

**Design doc:** `docs/plans/2026-02-22-llm-filter-rank-design.md`

---

### Task 1: Update data models (`models.py`)

**Files:**
- Modify: `src/paperdigest/models.py`
- Test: `tests/test_models.py` (new file)

**Step 1: Write the failing test**

Create `tests/test_models.py`:

```python
"""Tests for data models."""

from datetime import datetime, timezone
from paperdigest.models import FilterResult, Paper, Scores


def _make_paper(**kwargs):
    defaults = dict(
        arxiv_id="2401.00001",
        title="Test Paper",
        abstract="An abstract.",
        authors=["Author A"],
        published=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return Paper(**defaults)


class TestScores:
    def test_scores_has_quality_and_llm_rank(self):
        s = Scores(quality=0.8, llm_rank=3)
        assert s.quality == 0.8
        assert s.llm_rank == 3

    def test_scores_defaults(self):
        s = Scores()
        assert s.quality == 0.0
        assert s.llm_rank == 0


class TestFilterResult:
    def test_filter_result_fields(self):
        paper = _make_paper()
        fr = FilterResult(paper=paper, relevant=True, reason="On topic")
        assert fr.relevant is True
        assert fr.reason == "On topic"
        assert fr.paper.arxiv_id == "2401.00001"

    def test_filter_result_rejected(self):
        paper = _make_paper()
        fr = FilterResult(paper=paper, relevant=False, reason="Off topic")
        assert fr.relevant is False
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'FilterResult'` and `Scores` constructor mismatch

**Step 3: Write minimal implementation**

In `src/paperdigest/models.py`, make these changes:

1. Replace the `Scores` dataclass (lines 33-39):
```python
@dataclass
class Scores:
    """Computed scores for a paper."""

    quality: float = 0.0
    llm_rank: int = 0
```

2. Add `FilterResult` after `Summary` (after line 53):
```python
@dataclass
class FilterResult:
    """Result of LLM relevance filtering for a paper."""

    paper: Paper
    relevant: bool
    reason: str = ""
```

3. Update `Digest` to include rejected papers (replace lines 66-74):
```python
@dataclass
class Digest:
    """A complete digest ready for delivery."""

    date: datetime
    topic_name: str
    entries: list[DigestEntry] = field(default_factory=list)
    rejected: list[FilterResult] = field(default_factory=list)
    total_collected: int = 0
    total_new: int = 0
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add tests/test_models.py src/paperdigest/models.py
git commit -m "refactor: update Scores model, add FilterResult dataclass"
```

---

### Task 2: Update config dataclasses (`config.py`)

**Files:**
- Modify: `src/paperdigest/config.py`
- Modify: `tests/test_config.py`

**Step 1: Write the failing tests**

Add to `tests/test_config.py` (these replace/update existing tests):

```python
class TestNewConfig:
    def test_topic_description_loads(self, tmp_path):
        data = {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
                "description": "Papers about testing frameworks and CI/CD pipelines.",
            }
        }
        cfg = load_config(_write_config(tmp_path, data))
        assert cfg.topic.description == "Papers about testing frameworks and CI/CD pipelines."

    def test_topic_description_defaults_empty(self, tmp_path):
        data = {"topic": {"name": "Test", "primary_keywords": ["test"]}}
        cfg = load_config(_write_config(tmp_path, data))
        assert cfg.topic.description == ""

    def test_split_llm_config(self, tmp_path):
        data = {
            "topic": {"name": "Test", "primary_keywords": ["test"]},
            "llm": {
                "filter": {
                    "enabled": True,
                    "model": "gpt-4o-mini",
                    "cost_control": {"max_cost_per_run": 0.10, "max_cost_per_month": 3.00},
                },
                "summarizer": {
                    "enabled": True,
                    "model": "gpt-5-nano",
                    "max_text_chars": 40000,
                    "cost_control": {"max_cost_per_run": 0.50, "max_cost_per_month": 10.00},
                },
            },
        }
        cfg = load_config(_write_config(tmp_path, data))
        assert cfg.llm.filter.enabled is True
        assert cfg.llm.filter.model == "gpt-4o-mini"
        assert cfg.llm.filter.cost_control.max_cost_per_run == 0.10
        assert cfg.llm.summarizer.enabled is True
        assert cfg.llm.summarizer.model == "gpt-5-nano"
        assert cfg.llm.summarizer.max_text_chars == 40000
        assert cfg.llm.summarizer.cost_control.max_cost_per_month == 10.00

    def test_llm_defaults_both_disabled(self, tmp_path):
        data = {"topic": {"name": "Test", "primary_keywords": ["test"]}}
        cfg = load_config(_write_config(tmp_path, data))
        assert cfg.llm.filter.enabled is False
        assert cfg.llm.summarizer.enabled is False

    def test_scoring_no_alpha_no_relevance(self, tmp_path):
        data = {
            "topic": {"name": "Test", "primary_keywords": ["test"]},
            "scoring": {
                "quality": {
                    "w_venue": 0.25,
                    "w_author": 0.20,
                    "w_cite": 0.20,
                    "w_code": 0.15,
                    "w_fresh": 0.20,
                },
            },
        }
        cfg = load_config(_write_config(tmp_path, data))
        # ScoringConfig should have quality and venue_tiers, no alpha or relevance
        assert hasattr(cfg.scoring, "quality")
        assert hasattr(cfg.scoring, "venue_tiers")
        assert not hasattr(cfg.scoring, "alpha")
        assert not hasattr(cfg.scoring, "relevance")
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py::TestNewConfig -v`
Expected: FAIL — `topic.description` doesn't exist, `llm.filter` doesn't exist

**Step 3: Implement config changes**

In `src/paperdigest/config.py`:

1. Add `description` to `TopicConfig` (line 21):
```python
@dataclass
class TopicConfig:
    name: str
    primary_keywords: list[str]
    secondary_keywords: list[str] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=list)
    arxiv_categories: list[str] = field(default_factory=list)
    description: str = ""
```

2. Remove `RelevanceWeights` (lines 45-51) entirely.

3. Remove `alpha` and `relevance` from `ScoringConfig` (lines 63-68):
```python
@dataclass
class ScoringConfig:
    quality: QualityWeights = field(default_factory=QualityWeights)
    venue_tiers: dict[str, list[str]] = field(default_factory=dict)
```

4. Split `LLMConfig` into `FilterLLMConfig` and `SummarizerLLMConfig`:
```python
@dataclass
class FilterLLMConfig:
    enabled: bool = False
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float | None = None
    max_completion_tokens: int | None = 256
    cost_control: CostControl = field(default_factory=CostControl)


@dataclass
class SummarizerLLMConfig:
    enabled: bool = False
    model: str = "gpt-5-nano-2025-08-07"
    base_url: str = "https://api.openai.com/v1"
    temperature: float | None = None
    max_completion_tokens: int | None = 16384
    max_text_chars: int = 50000
    cost_control: CostControl = field(default_factory=CostControl)


@dataclass
class LLMConfig:
    filter: FilterLLMConfig = field(default_factory=FilterLLMConfig)
    summarizer: SummarizerLLMConfig = field(default_factory=SummarizerLLMConfig)
```

5. Update `_build_topic` to include `description`:
```python
def _build_topic(d: dict) -> TopicConfig:
    return TopicConfig(
        name=d["name"],
        primary_keywords=d["primary_keywords"],
        secondary_keywords=d.get("secondary_keywords", []),
        benchmarks=d.get("benchmarks", []),
        arxiv_categories=d.get("arxiv_categories", []),
        description=d.get("description", ""),
    )
```

6. Remove `_build_scoring`'s relevance and alpha handling. Replace with:
```python
def _build_scoring(d: dict) -> ScoringConfig:
    qual = d.get("quality", {})
    return ScoringConfig(
        quality=QualityWeights(**qual),
        venue_tiers=d.get("venue_tiers", {}),
    )
```

7. Replace `_build_llm` with:
```python
def _build_llm(d: dict) -> LLMConfig:
    filter_raw = d.get("filter", {})
    summ_raw = d.get("summarizer", {})

    filter_cc = filter_raw.get("cost_control", {})
    summ_cc = summ_raw.get("cost_control", {})

    return LLMConfig(
        filter=FilterLLMConfig(
            enabled=filter_raw.get("enabled", False),
            model=filter_raw.get("model", "gpt-4o-mini"),
            base_url=filter_raw.get("base_url", "https://api.openai.com/v1"),
            temperature=filter_raw.get("temperature"),
            max_completion_tokens=filter_raw.get("max_completion_tokens", 256),
            cost_control=CostControl(**filter_cc),
        ),
        summarizer=SummarizerLLMConfig(
            enabled=summ_raw.get("enabled", False),
            model=summ_raw.get("model", "gpt-5-nano-2025-08-07"),
            base_url=summ_raw.get("base_url", "https://api.openai.com/v1"),
            temperature=summ_raw.get("temperature"),
            max_completion_tokens=summ_raw.get("max_completion_tokens", 16384),
            max_text_chars=summ_raw.get("max_text_chars", 50000),
            cost_control=CostControl(**summ_cc),
        ),
    )
```

8. In `load_config`, remove the `alpha` validation (lines 276-277) and remove the `RelevanceWeights` import reference. Keep the quality weights validation.

9. Update `Config.llm_api_key` — no changes needed, it stays the same env var.

**Step 4: Fix existing config tests**

Update existing tests in `tests/test_config.py`:
- `test_valid_config`: Remove assertion on `scoring.alpha == 0.65`
- `test_custom_scoring`: Remove `alpha` from config data and assertions. Remove `relevance` section.
- `test_llm_defaults_disabled`: Change from `cfg.llm.enabled` to `cfg.llm.filter.enabled` and `cfg.llm.summarizer.enabled`
- `test_llm_cost_control`: Update to use `cfg.llm.summarizer.cost_control.*`
- `test_invalid_alpha_raises`: Delete this test entirely
- `test_invalid_quality_weights_raises`: Keep as-is (quality validation unchanged)

**Step 5: Run all config tests**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/paperdigest/config.py tests/test_config.py
git commit -m "refactor: split LLM config into filter + summarizer, remove keyword relevance config"
```

---

### Task 3: Update database schema and methods (`db.py`)

**Files:**
- Modify: `src/paperdigest/db.py`
- Test: existing DB usage in other tests (no dedicated DB test file)

**Step 1: Write a failing test**

Add `tests/test_db.py`:

```python
"""Tests for database layer."""

import pytest
from datetime import datetime, timezone
from paperdigest.db import Database
from paperdigest.models import FilterResult, Paper, Scores


def _make_paper(**kwargs):
    defaults = dict(
        arxiv_id="2401.00001",
        title="Test Paper",
        abstract="An abstract.",
        authors=["Author A"],
        published=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return Paper(**defaults)


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "test.db") as d:
        d.init_schema()
        yield d


class TestFilterResults:
    def test_upsert_and_get_filter_results(self, db):
        paper = _make_paper()
        paper_id = db.upsert_paper(paper)
        db.upsert_filter_result(paper_id, relevant=True, reason="On topic")
        results = db.get_filter_results()
        assert len(results) == 1
        assert results[0]["relevant"] == 1
        assert results[0]["reason"] == "On topic"

    def test_get_rejected_papers(self, db):
        p1 = _make_paper(arxiv_id="2401.00001", title="Good paper")
        p2 = _make_paper(arxiv_id="2401.00002", title="Bad paper")
        id1 = db.upsert_paper(p1)
        id2 = db.upsert_paper(p2)
        db.upsert_filter_result(id1, relevant=True, reason="On topic")
        db.upsert_filter_result(id2, relevant=False, reason="Off topic")
        rejected = db.get_rejected_papers()
        assert len(rejected) == 1
        assert rejected[0][0].arxiv_id == "2401.00002"
        assert rejected[0][1] == "Off topic"


class TestUpdatedScores:
    def test_upsert_scores_with_llm_rank(self, db):
        paper = _make_paper()
        paper_id = db.upsert_paper(paper)
        scores = Scores(quality=0.8, llm_rank=3)
        db.upsert_scores(paper_id, scores)
        top = db.get_top_scored_papers(limit=1)
        assert len(top) == 1
        _, s = top[0]
        assert s.quality == pytest.approx(0.8)
        assert s.llm_rank == 3
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_db.py -v`
Expected: FAIL — `upsert_filter_result` doesn't exist, `scores` table has wrong columns

**Step 3: Update database schema and methods**

In `src/paperdigest/db.py`:

1. Update `SCHEMA` — replace the scores table definition (lines 34-40):
```sql
CREATE TABLE IF NOT EXISTS scores (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id),
    quality REAL NOT NULL,
    llm_rank INTEGER NOT NULL DEFAULT 0,
    scored_at TEXT DEFAULT (datetime('now'))
);
```

2. Add `paper_filter_results` table to `SCHEMA` (after the llm_usage table, before indexes):
```sql
CREATE TABLE IF NOT EXISTS paper_filter_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    run_date TEXT NOT NULL,
    relevant INTEGER NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);
```

3. Add index for filter results (with other indexes):
```sql
CREATE INDEX IF NOT EXISTS idx_filter_results_paper_date ON paper_filter_results(paper_id, run_date);
```

4. Update `upsert_scores` method (lines 219-230):
```python
def upsert_scores(self, paper_id: int, scores: Scores):
    self.conn.execute(
        """INSERT INTO scores (paper_id, quality, llm_rank, scored_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(paper_id) DO UPDATE SET
            quality=excluded.quality,
            llm_rank=excluded.llm_rank,
            scored_at=datetime('now')""",
        (paper_id, scores.quality, scores.llm_rank),
    )
    self.conn.commit()
```

5. Update `get_top_scored_papers` (lines 232-250):
```python
def get_top_scored_papers(self, limit: int = 20) -> list[tuple[Paper, Scores]]:
    rows = self.conn.execute(
        """SELECT p.*, s.quality, s.llm_rank
        FROM papers p
        JOIN scores s ON p.id = s.paper_id
        ORDER BY s.llm_rank ASC, s.quality DESC
        LIMIT ?""",
        (limit,),
    ).fetchall()
    results = []
    for row in rows:
        paper = self._row_to_paper(row)
        scores = Scores(
            quality=row["quality"],
            llm_rank=row["llm_rank"],
        )
        results.append((paper, scores))
    return results
```

6. Add new methods for filter results:
```python
def upsert_filter_result(self, paper_id: int, relevant: bool, reason: str):
    """Store a filter result for a paper."""
    self.conn.execute(
        """INSERT INTO paper_filter_results (paper_id, run_date, relevant, reason)
        VALUES (?, date('now'), ?, ?)""",
        (paper_id, int(relevant), reason),
    )
    self.conn.commit()

def get_filter_results(self, run_date: str | None = None) -> list[dict]:
    """Get all filter results, optionally for a specific date."""
    if run_date:
        rows = self.conn.execute(
            "SELECT * FROM paper_filter_results WHERE run_date = ?", (run_date,)
        ).fetchall()
    else:
        rows = self.conn.execute("SELECT * FROM paper_filter_results").fetchall()
    return [dict(row) for row in rows]

def get_rejected_papers(self, run_date: str | None = None) -> list[tuple[Paper, str]]:
    """Get papers that were rejected by the filter, with their reasons."""
    query = """
        SELECT p.*, fr.reason
        FROM paper_filter_results fr
        JOIN papers p ON fr.paper_id = p.id
        WHERE fr.relevant = 0
    """
    params = ()
    if run_date:
        query += " AND fr.run_date = ?"
        params = (run_date,)
    query += " ORDER BY p.published DESC"
    rows = self.conn.execute(query, params).fetchall()
    return [(self._row_to_paper(row), row["reason"]) for row in rows]
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/paperdigest/db.py tests/test_db.py
git commit -m "refactor: update scores schema, add paper_filter_results table"
```

---

### Task 4: Refactor scoring to quality-only (`scoring.py`)

**Files:**
- Modify: `src/paperdigest/scoring.py`
- Modify: `tests/test_scoring.py`

**Step 1: Write updated scoring tests**

Rewrite `tests/test_scoring.py`. Remove all `TestRelevanceScoring` tests. Remove `TestRanking` (ranking is now LLM-driven). Keep and update `TestQualityScoring`:

```python
"""Tests for quality scoring."""

from datetime import datetime, timedelta, timezone

import pytest

from paperdigest.config import Config, QualityWeights, ScoringConfig, TopicConfig
from paperdigest.models import Paper, Scores
from paperdigest.scoring import compute_quality, score_papers


REF_TIME = datetime.now(timezone.utc)


def _make_config(**overrides):
    scoring_data = {
        "quality": QualityWeights(
            w_venue=0.25, w_author=0.20, w_cite=0.20, w_code=0.15, w_fresh=0.20
        ),
        "venue_tiers": {
            "tier1": ["CVPR", "NeurIPS", "ICML", "ICLR"],
            "tier2": ["IROS", "ICRA"],
            "tier3": ["IV", "ITSC"],
        },
    }
    scoring_data.update(overrides.pop("scoring", {}))
    return Config(
        topic=TopicConfig(name="Test", primary_keywords=["test"]),
        scoring=ScoringConfig(**scoring_data),
        **overrides,
    )


def _make_paper(**kwargs):
    defaults = dict(
        arxiv_id="2401.00001",
        title="Test Paper",
        abstract="An abstract.",
        authors=["Author A"],
        published=REF_TIME,
    )
    defaults.update(kwargs)
    return Paper(**defaults)


class TestQualityScoring:
    def test_high_quality_paper(self):
        config = _make_config()
        paper = _make_paper(
            citations=100, max_hindex=60, venue="CVPR 2024",
            code_url="https://github.com/test", published=REF_TIME,
        )
        q = compute_quality(paper, config.scoring)
        assert q > 0.7

    def test_low_quality_paper(self):
        config = _make_config()
        paper = _make_paper(
            citations=0, max_hindex=2,
            published=REF_TIME - timedelta(days=25),
        )
        q = compute_quality(paper, config.scoring)
        assert q < 0.5

    def test_freshness_decay(self):
        config = _make_config()
        fresh = _make_paper(published=REF_TIME)
        stale = _make_paper(published=REF_TIME - timedelta(days=25))
        q_fresh = compute_quality(fresh, config.scoring)
        q_stale = compute_quality(stale, config.scoring)
        assert q_fresh > q_stale


class TestScorePapers:
    def test_returns_papers_with_quality_scores(self):
        config = _make_config()
        papers = [
            _make_paper(arxiv_id="1", citations=100, venue="CVPR 2024"),
            _make_paper(arxiv_id="2", citations=0),
        ]
        scored = score_papers(papers, config)
        assert len(scored) == 2
        for paper, scores in scored:
            assert scores.quality >= 0.0
            assert scores.llm_rank == 0  # not ranked yet
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: FAIL — `score_papers` returns wrong types, `Scores` constructor mismatch

**Step 3: Rewrite scoring.py**

Replace `src/paperdigest/scoring.py` entirely:

```python
"""Paper scoring: quality signals only (relevance handled by LLM filter)."""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone

from .config import Config, QualityWeights, ScoringConfig
from .models import Paper, Scores

logger = logging.getLogger(__name__)


def _venue_tier_score(venue: str | None, venue_tiers: dict[str, list[str]]) -> float:
    """Map venue to tier score using word-boundary matching."""
    if not venue:
        return 0.2
    for tier, tier_score in [("tier1", 1.0), ("tier2", 0.7), ("tier3", 0.4)]:
        for v in venue_tiers.get(tier, []):
            if re.search(r'\b' + re.escape(v) + r'\b', venue, re.IGNORECASE):
                return tier_score
    return 0.2


def compute_quality(paper: Paper, config: ScoringConfig) -> float:
    """Compute quality score (0-1) as weighted sum of signals."""
    qw: QualityWeights = config.quality

    venue_score = _venue_tier_score(paper.venue, config.venue_tiers)
    author_score = min(1.0, (paper.max_hindex or 0) / 50.0)
    cite_score = min(1.0, math.log(1 + (paper.citations or 0)) / 5.0)
    code_score = 1.0 if paper.code_url else 0.0

    published = (
        paper.published.astimezone(timezone.utc)
        if paper.published.tzinfo
        else paper.published.replace(tzinfo=timezone.utc)
    )
    age_days = (datetime.now(timezone.utc) - published).days
    fresh_score = max(0.0, 1.0 - age_days / 30.0)

    score = (
        qw.w_venue * venue_score
        + qw.w_author * author_score
        + qw.w_cite * cite_score
        + qw.w_code * code_score
        + qw.w_fresh * fresh_score
    )

    return min(score, 1.0)


def score_papers(papers: list[Paper], config: Config) -> list[tuple[Paper, Scores]]:
    """Compute quality scores for all papers. Returns list of (Paper, Scores).

    Note: llm_rank is set to 0 here — ranking is done by the summarizer.
    """
    results = []
    for paper in papers:
        quality = compute_quality(paper, config.scoring)
        scores = Scores(quality=quality, llm_rank=0)
        results.append((paper, scores))

    results.sort(key=lambda x: x[1].quality, reverse=True)
    logger.info(f"Scored {len(results)} papers (quality only)")
    if results:
        top = results[0]
        logger.info(f"Top quality: [{top[1].quality:.3f}] {top[0].title[:70]}")
    return results
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/paperdigest/scoring.py tests/test_scoring.py
git commit -m "refactor: remove keyword relevance scoring, keep quality-only scoring"
```

---

### Task 5: Create LLM filter module (`filter.py`)

**Files:**
- Create: `src/paperdigest/filter.py`
- Create: `tests/test_filter.py`

**Step 1: Write the failing tests**

Create `tests/test_filter.py`:

```python
"""Tests for LLM relevance filter."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from paperdigest.config import (
    Config, CostControl, FilterLLMConfig, LLMConfig, SummarizerLLMConfig, TopicConfig,
)
from paperdigest.db import Database
from paperdigest.filter import PaperFilter
from paperdigest.models import Paper


def _make_paper(arxiv_id="2401.00001", title="VLM for Driving", abstract="We present a VLM."):
    return Paper(
        arxiv_id=arxiv_id, title=title, abstract=abstract,
        authors=["Author A"], published=datetime.now(timezone.utc),
    )


def _make_config(**overrides):
    return Config(
        topic=TopicConfig(
            name="Test",
            primary_keywords=["test"],
            description="Papers about vision-language models for autonomous driving.",
        ),
        llm=LLMConfig(
            filter=FilterLLMConfig(
                enabled=True,
                model="gpt-4o-mini",
                cost_control=CostControl(
                    max_cost_per_run=0.10,
                    max_cost_per_month=3.00,
                    input_cost_per_1k=0.00015,
                    output_cost_per_1k=0.0006,
                ),
            ),
        ),
        **overrides,
    )


def _make_llm_response(content: str, input_tokens: int = 200, output_tokens: int = 50):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = input_tokens
    resp.usage.completion_tokens = output_tokens
    return resp


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "test.db") as d:
        d.init_schema()
        yield d


class TestFilterPaper:
    @patch("paperdigest.filter.OpenAI")
    def test_relevant_paper(self, mock_openai_cls, db):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        response_json = json.dumps({"relevant": True, "reason": "Directly about VLMs for driving"})
        mock_client.chat.completions.create.return_value = _make_llm_response(response_json)

        config = _make_config()
        filt = PaperFilter(config, db)
        filt._client = mock_client
        paper = _make_paper()
        result = filt.filter_paper(paper)

        assert result.relevant is True
        assert "VLM" in result.reason

    @patch("paperdigest.filter.OpenAI")
    def test_irrelevant_paper(self, mock_openai_cls, db):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        response_json = json.dumps({"relevant": False, "reason": "About NLP translation, not driving"})
        mock_client.chat.completions.create.return_value = _make_llm_response(response_json)

        config = _make_config()
        filt = PaperFilter(config, db)
        filt._client = mock_client
        paper = _make_paper(title="Machine Translation", abstract="We translate text.")
        result = filt.filter_paper(paper)

        assert result.relevant is False

    @patch("paperdigest.filter.OpenAI")
    def test_invalid_json_fails_open(self, mock_openai_cls, db):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response("not valid json")

        config = _make_config()
        filt = PaperFilter(config, db)
        filt._client = mock_client
        paper = _make_paper()
        result = filt.filter_paper(paper)

        # Fail-open: treat as relevant
        assert result.relevant is True

    @patch("paperdigest.filter.OpenAI")
    def test_api_error_fails_open(self, mock_openai_cls, db):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        config = _make_config()
        filt = PaperFilter(config, db)
        filt._client = mock_client
        paper = _make_paper()
        result = filt.filter_paper(paper)

        assert result.relevant is True


class TestFilterPapers:
    @patch("paperdigest.filter.OpenAI")
    def test_splits_relevant_and_rejected(self, mock_openai_cls, db):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        responses = [
            _make_llm_response(json.dumps({"relevant": True, "reason": "On topic"})),
            _make_llm_response(json.dumps({"relevant": False, "reason": "Off topic"})),
            _make_llm_response(json.dumps({"relevant": True, "reason": "Related"})),
        ]
        mock_client.chat.completions.create.side_effect = responses

        config = _make_config()
        filt = PaperFilter(config, db)
        filt._client = mock_client

        papers = [
            _make_paper(arxiv_id="1", title="VLM Driving"),
            _make_paper(arxiv_id="2", title="NLP Translation"),
            _make_paper(arxiv_id="3", title="Driving Planning"),
        ]
        # Store papers in DB so they have db_ids
        for p in papers:
            p.db_id = db.upsert_paper(p)

        relevant, rejected = filt.filter_papers(papers)
        assert len(relevant) == 2
        assert len(rejected) == 1
        assert rejected[0].paper.arxiv_id == "2"


class TestFilterBudget:
    @patch("paperdigest.filter.OpenAI")
    def test_budget_exhaustion_fails_open(self, mock_openai_cls, db):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        config = _make_config()
        # Override with tiny budget
        config.llm.filter.cost_control.max_cost_per_run = 0.0001

        filt = PaperFilter(config, db)
        filt._client = mock_client
        # Simulate having already spent the budget
        filt.run_cost = 0.0001
        filt.run_papers = 1

        paper = _make_paper()
        result = filt.filter_paper(paper)

        # Fail-open: treat as relevant when budget exhausted
        assert result.relevant is True
        # API should NOT have been called
        mock_client.chat.completions.create.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_filter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paperdigest.filter'`

**Step 3: Implement the filter module**

Create `src/paperdigest/filter.py`:

```python
"""LLM-based paper relevance filtering."""

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
    """LLM-based paper filter with cost tracking and fail-open behavior."""

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
        cc = self.config.llm.filter.cost_control

        monthly_cost = self.db.get_monthly_llm_cost()
        if monthly_cost >= cc.max_cost_per_month:
            return False, f"Monthly filter budget exhausted (${monthly_cost:.2f})"

        if self.run_papers > 0:
            avg_cost = self.run_cost / self.run_papers
            estimated_next = avg_cost * 1.5
        else:
            estimated_next = self._estimate_cost(300, 50)
        if self.run_cost + estimated_next > cc.max_cost_per_run:
            return False, f"Per-run filter budget would be exceeded (${self.run_cost:.4f})"

        return True, ""

    def filter_paper(self, paper: Paper) -> FilterResult:
        """Filter a single paper. Fails open (returns relevant=True) on any error."""
        ok, reason = self._check_budget()
        if not ok:
            logger.warning(f"Filter budget exceeded for {paper.arxiv_id}: {reason}")
            return FilterResult(paper=paper, relevant=True, reason=f"Budget exceeded: {reason}")

        system = FILTER_SYSTEM_PROMPT.format(description=self.config.topic.description)
        user = FILTER_USER_TEMPLATE.format(title=paper.title, abstract=paper.abstract)

        try:
            kwargs = dict(
                model=self.config.llm.filter.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            if self.config.llm.filter.temperature is not None:
                kwargs["temperature"] = self.config.llm.filter.temperature
            if self.config.llm.filter.max_completion_tokens is not None:
                kwargs["max_completion_tokens"] = self.config.llm.filter.max_completion_tokens

            response = self.client.chat.completions.create(**kwargs)

            usage = response.usage
            if usage:
                input_t = usage.prompt_tokens
                output_t = usage.completion_tokens
                cost = self._estimate_cost(input_t, output_t)
                self.run_input_tokens += input_t
                self.run_output_tokens += output_t
                self.run_cost += cost

            if not response.choices or response.choices[0].message.content is None:
                logger.warning(f"Empty filter response for {paper.arxiv_id}")
                return FilterResult(paper=paper, relevant=True, reason="Empty LLM response")

            content = response.choices[0].message.content.strip()
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
            logger.warning(f"Failed to parse filter JSON for {paper.arxiv_id}, treating as relevant")
            return FilterResult(paper=paper, relevant=True, reason="JSON parse error")
        except Exception:
            logger.exception(f"Filter LLM call failed for {paper.arxiv_id}, treating as relevant")
            return FilterResult(paper=paper, relevant=True, reason="LLM error")

    def filter_papers(self, papers: list[Paper]) -> tuple[list[Paper], list[FilterResult]]:
        """Filter multiple papers. Returns (relevant_papers, rejected_results)."""
        relevant = []
        rejected = []

        for i, paper in enumerate(papers):
            logger.info(f"Filtering [{i+1}/{len(papers)}] {paper.arxiv_id}: {paper.title[:60]}")
            result = self.filter_paper(paper)

            if paper.db_id:
                self.db.upsert_filter_result(paper.db_id, result.relevant, result.reason)

            if result.relevant:
                relevant.append(paper)
            else:
                rejected.append(result)

        # Log usage
        if self.run_papers > 0:
            self.db.log_llm_usage(
                run_id=self.run_id,
                papers_summarized=self.run_papers,
                input_tokens=self.run_input_tokens,
                output_tokens=self.run_output_tokens,
                estimated_cost=self.run_cost,
            )

        logger.info(
            f"Filter complete: {len(relevant)} relevant, {len(rejected)} rejected "
            f"(${self.run_cost:.4f} estimated cost)"
        )
        return relevant, rejected
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_filter.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/paperdigest/filter.py tests/test_filter.py
git commit -m "feat: add LLM-based paper relevance filter module"
```

---

### Task 6: Refactor summarizer — always full text + ranking (`summarizer.py`)

**Files:**
- Modify: `src/paperdigest/summarizer.py`
- Modify: `tests/test_summarizer.py`

**Step 1: Update summarizer tests**

Update `tests/test_summarizer.py` to reflect:
1. Config changes (use `cfg.llm.summarizer.*` instead of `cfg.llm.*`)
2. Always full text (no `use_full_text` toggle)
3. New `rank_papers` method

Key test changes to `_make_config`:
```python
def _make_config(**overrides):
    return Config(
        topic=TopicConfig(
            name="AD",
            primary_keywords=["driving"],
            description="Papers about autonomous driving.",
        ),
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
        **overrides,
    )
```

Add new test class for ranking:
```python
class TestRanking:
    @patch("paperdigest.summarizer.OpenAI")
    def test_rank_papers(self, mock_openai_cls, db):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        ranking_json = json.dumps({"ranking": ["2401.00002", "2401.00001"]})
        mock_client.chat.completions.create.return_value = _make_llm_response(ranking_json)

        config = _make_config()
        summarizer = Summarizer(config, db)
        summarizer._client = mock_client

        papers = [
            _make_paper("2401.00001", "Paper A"),
            _make_paper("2401.00002", "Paper B"),
        ]
        summaries = {
            "2401.00001": Summary(one_liner="Method A"),
            "2401.00002": Summary(one_liner="Method B"),
        }
        quality_scores = {"2401.00001": 0.7, "2401.00002": 0.5}

        ranking = summarizer.rank_papers(papers, summaries, quality_scores)
        assert ranking == {"2401.00002": 1, "2401.00001": 2}
```

Update all existing tests to use the new config structure (replace `LLMConfig(enabled=True, ...)` with the nested `LLMConfig(summarizer=SummarizerLLMConfig(enabled=True, ...))`).

Update `_build_messages` tests — since it always tries full text now, mock `fetch_paper_text` to return `None` so it falls back to abstract for tests that don't care about PDF.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_summarizer.py -v`
Expected: FAIL — config structure mismatch

**Step 3: Refactor summarizer**

In `src/paperdigest/summarizer.py`:

1. Update `__init__` to use `config.llm.summarizer`:
```python
def __init__(self, config: Config, db: Database):
    self.config = config
    self.llm_cfg = config.llm.summarizer
    self.db = db
    self.run_id = str(uuid.uuid4())[:8]
    self.run_cost = 0.0
    self.run_input_tokens = 0
    self.run_output_tokens = 0
    self.run_papers = 0
    self._client = None
```

2. Update `client` property to use `self.llm_cfg.base_url`.

3. Update `_estimate_cost` to use `self.llm_cfg.cost_control`.

4. Update `_check_budget` to use `self.llm_cfg.cost_control`.

5. Update `_build_messages` — always try to fetch full text (remove `use_full_text` check):
```python
def _build_messages(self, paper: Paper) -> list[dict]:
    full_text = None
    pdf_url = paper.oa_pdf_url or paper.pdf_url
    if pdf_url:
        from .pdf import fetch_paper_text
        logger.info(f"Fetching full text for {paper.arxiv_id}...")
        full_text = fetch_paper_text(pdf_url, max_chars=self.llm_cfg.max_text_chars)
        if full_text:
            logger.info(f"Extracted {len(full_text)} chars from PDF")
        else:
            logger.warning(f"Full text extraction failed for {paper.arxiv_id}, falling back to abstract")
    else:
        logger.warning(f"No PDF URL for {paper.arxiv_id}, falling back to abstract")

    if full_text:
        system = SYSTEM_PROMPT_FULL_TEXT
        user = USER_TEMPLATE_FULL_TEXT.format(title=paper.title, full_text=full_text)
    else:
        system = SYSTEM_PROMPT_ABSTRACT
        user = USER_TEMPLATE_ABSTRACT.format(title=paper.title, abstract=paper.abstract)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
```

6. Update `summarize_paper` to use `self.llm_cfg` for model, temperature, max_completion_tokens.

7. Add `rank_papers` method:
```python
RANKING_SYSTEM_PROMPT = """You are ranking research papers by relevance and importance for a digest about:
{description}

You will receive a list of papers with their summaries and quality metadata.
Rank them from most to least relevant/important to the described topic.
Respond with ONLY valid JSON: {{"ranking": ["arxiv_id_1", "arxiv_id_2", ...]}}"""

RANKING_USER_TEMPLATE = """Papers to rank:

{papers_list}"""

def rank_papers(
    self,
    papers: list[Paper],
    summaries: dict[str, Summary],
    quality_scores: dict[str, float],
) -> dict[str, int]:
    """Rank papers by LLM. Returns {arxiv_id: rank} where 1=best."""
    if not papers:
        return {}

    # Build papers list for the prompt
    lines = []
    for p in papers:
        summary = summaries.get(p.arxiv_id)
        one_liner = summary.one_liner if summary else "(no summary)"
        quality = quality_scores.get(p.arxiv_id, 0.0)
        line = (
            f"- ID: {p.arxiv_id} | Title: {p.title} | "
            f"Summary: {one_liner} | "
            f"Venue: {p.venue or 'N/A'} | Citations: {p.citations or 0} | "
            f"Quality: {quality:.2f}"
        )
        lines.append(line)

    system = RANKING_SYSTEM_PROMPT.format(description=self.config.topic.description)
    user = RANKING_USER_TEMPLATE.format(papers_list="\n".join(lines))

    try:
        kwargs = dict(
            model=self.llm_cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self.llm_cfg.temperature is not None:
            kwargs["temperature"] = self.llm_cfg.temperature

        response = self.client.chat.completions.create(**kwargs)

        usage = response.usage
        if usage:
            cost = self._estimate_cost(usage.prompt_tokens, usage.completion_tokens)
            self.run_input_tokens += usage.prompt_tokens
            self.run_output_tokens += usage.completion_tokens
            self.run_cost += cost

        if not response.choices or response.choices[0].message.content is None:
            logger.warning("Empty ranking response, using quality-based order")
            return self._fallback_ranking(papers, quality_scores)

        content = response.choices[0].message.content.strip()
        content = re.sub(r'^```\w*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        data = json.loads(content.strip())

        ranking = {}
        for rank, arxiv_id in enumerate(data["ranking"], 1):
            ranking[arxiv_id] = rank

        # Add any papers missing from LLM response at the end
        next_rank = len(ranking) + 1
        for p in papers:
            if p.arxiv_id not in ranking:
                ranking[p.arxiv_id] = next_rank
                next_rank += 1

        return ranking

    except (json.JSONDecodeError, KeyError, Exception):
        logger.exception("Ranking LLM call failed, falling back to quality-based order")
        return self._fallback_ranking(papers, quality_scores)

def _fallback_ranking(self, papers: list[Paper], quality_scores: dict[str, float]) -> dict[str, int]:
    """Rank by quality score when LLM ranking fails."""
    sorted_papers = sorted(papers, key=lambda p: quality_scores.get(p.arxiv_id, 0.0), reverse=True)
    return {p.arxiv_id: rank for rank, p in enumerate(sorted_papers, 1)}
```

**Step 4: Run all summarizer tests**

Run: `python -m pytest tests/test_summarizer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/paperdigest/summarizer.py tests/test_summarizer.py
git commit -m "refactor: summarizer always uses full text, add LLM ranking"
```

---

### Task 7: Update CLI pipeline (`cli.py`)

**Files:**
- Modify: `src/paperdigest/cli.py`

**Step 1: Update `cmd_score` — quality only**

Replace lines 138-155:
```python
def cmd_score(args):
    """Score all papers (quality signals only)."""
    config = get_config(args)
    with Database(config.db_path) as db:
        db.init_schema()

        from .scoring import score_papers

        papers = db.get_all_papers()
        if not papers:
            logger.info("No papers to score")
            return

        scored = score_papers(papers, config)
        for paper, scores in scored:
            db.upsert_scores(paper.db_id, scores)

        logger.info(f"Scored {len(scored)} papers (quality)")
```

**Step 2: Add `cmd_filter`**

Add after `cmd_fetch`:
```python
def cmd_filter(args):
    """Filter papers using LLM relevance check."""
    config = get_config(args)
    if not config.llm.filter.enabled:
        logger.info("LLM filter is disabled in config")
        return

    with Database(config.db_path) as db:
        db.init_schema()

        from .filter import PaperFilter

        papers = db.get_all_papers()
        if not papers:
            logger.info("No papers to filter")
            return

        filt = PaperFilter(config, db)
        relevant, rejected = filt.filter_papers(papers)
        logger.info(f"Filter: {len(relevant)} relevant, {len(rejected)} rejected")
```

**Step 3: Update `cmd_digest` to include filtering, ranking, and rejected papers**

Replace `cmd_digest` (lines 158-212):
```python
def cmd_digest(args):
    """Generate and deliver digest."""
    config = get_config(args)
    with Database(config.db_path) as db:
        db.init_schema()

        dry_run = getattr(args, "dry_run", False)

        # Get all papers — filtering decides which are relevant
        papers = db.get_all_papers()
        if not papers:
            logger.info("No papers found. Run 'fetch' first.")
            return

        # LLM filter
        rejected_results = []
        if config.llm.filter.enabled and not dry_run:
            from .filter import PaperFilter

            filt = PaperFilter(config, db)
            papers, rejected_results = filt.filter_papers(papers)
            if not papers:
                logger.info("No relevant papers after filtering")
                return

        # Enrich survivors (if not already enriched)
        from .enrichment.pwc import enrich_with_pwc
        from .enrichment.semantic_scholar import enrich_papers

        unenriched = [p for p in papers if p.citations is None]
        if unenriched:
            logger.info(f"Enriching {len(unenriched)} papers...")
            unenriched = enrich_papers(unenriched, config)
            unenriched = enrich_with_pwc(unenriched, config)
            for paper in unenriched:
                db.update_enrichment(paper)
            # Refresh from DB to get enrichment data
            papers = [db.get_paper_by_arxiv_id(p.arxiv_id) for p in papers]
            papers = [p for p in papers if p is not None]

        # Quality scoring
        from .scoring import score_papers as compute_quality_scores

        scored = compute_quality_scores(papers, config)
        for paper, scores in scored:
            db.upsert_scores(paper.db_id, scores)

        # Limit to top_n for summarization
        top_papers = [p for p, _ in scored[:config.digest.top_n]]
        quality_map = {p.arxiv_id: s.quality for p, s in scored}

        # LLM summarize + rank
        summaries = {}
        ranking = {}
        if config.llm.summarizer.enabled and not dry_run:
            from .summarizer import Summarizer

            summarizer = Summarizer(config, db)
            summarize_subset = top_papers[:config.digest.summarize_top_n]
            summaries = summarizer.summarize_papers(summarize_subset)
            ranking = summarizer.rank_papers(top_papers, summaries, quality_map)
        elif dry_run:
            logger.info("Dry run — skipping LLM summarization and ranking")

        # If no LLM ranking, use quality order
        if not ranking:
            ranking = {p.arxiv_id: rank for rank, p in enumerate(top_papers, 1)}

        # Update scores with LLM rank
        for paper, scores in scored[:config.digest.top_n]:
            llm_rank = ranking.get(paper.arxiv_id, len(top_papers))
            scores.llm_rank = llm_rank
            db.upsert_scores(paper.db_id, scores)

        # Build digest entries sorted by LLM rank
        entries = []
        scored_map = {p.arxiv_id: (p, s) for p, s in scored}
        for paper in sorted(top_papers, key=lambda p: ranking.get(p.arxiv_id, 999)):
            p, s = scored_map[paper.arxiv_id]
            rank = ranking.get(p.arxiv_id, 0)
            summary = summaries.get(p.arxiv_id)
            entries.append(DigestEntry(paper=p, scores=s, rank=rank, summary=summary))

        total_papers = db.get_paper_count()
        from .models import FilterResult
        digest = Digest(
            date=datetime.now(timezone.utc),
            topic_name=config.topic.name,
            entries=entries,
            rejected=rejected_results,
            total_collected=total_papers,
            total_new=len(entries),
        )

        # Deliver
        from .delivery.markdown import deliver_markdown

        md_path = deliver_markdown(digest, config)
        logger.info(f"Markdown digest: {md_path}")

        tg_ok = True
        if config.delivery.telegram.enabled:
            from .delivery.telegram import deliver_telegram

            tg_ok = deliver_telegram(digest, config)

        paper_ids = [p.db_id for p in top_papers]
        status = "delivered" if tg_ok else "partial_delivery"
        db.log_digest(paper_ids, status=status)
```

**Step 4: Update `cmd_run` pipeline**

```python
def cmd_run(args):
    """Full pipeline: fetch -> digest (includes filter, enrich, score, summarize, rank)."""
    logger.info("Starting full pipeline run...")
    cmd_fetch(args)
    cmd_digest(args)
    logger.info("Pipeline run complete")
```

Note: `cmd_digest` now handles filter → enrich → score → summarize → rank internally. `cmd_enrich` and `cmd_score` remain as standalone commands for manual use.

**Step 5: Add `filter` subcommand to parser**

In the `main` function, add after the `score` parser:
```python
subparsers.add_parser("filter", parents=[common], help="Filter papers by LLM relevance")
```

And add to commands dict:
```python
"filter": cmd_filter,
```

**Step 6: Update `cmd_stats` for split LLM config**

Change line 249 from `if config.llm.enabled:` to:
```python
if config.llm.filter.enabled or config.llm.summarizer.enabled:
```

And change line 251 from `config.llm.cost_control.max_cost_per_month` to:
```python
monthly_budget = (
    config.llm.filter.cost_control.max_cost_per_month
    + config.llm.summarizer.cost_control.max_cost_per_month
)
```

**Step 7: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS (may need minor fixes to other test files that reference old config)

**Step 8: Commit**

```bash
git add src/paperdigest/cli.py
git commit -m "refactor: reorder pipeline with LLM filter, add filter subcommand"
```

---

### Task 8: Update digest template

**Files:**
- Modify: `src/paperdigest/templates/digest.md.j2`

**Step 1: Update scores display and add rejected section**

Replace the template:

```jinja2
# Paper Digest: {{ digest.topic_name }}

**Date:** {{ digest.date.strftime('%Y-%m-%d') }}
**Papers collected:** {{ digest.total_collected }} | **New:** {{ digest.total_new }} | **Ranked:** {{ digest.entries | length }}{% if digest.rejected %} | **Filtered out:** {{ digest.rejected | length }}{% endif %}

---

{% for entry in digest.entries %}
## {{ entry.rank }}. {{ entry.paper.title }}

| Field | Value |
|-------|-------|
| **arXiv** | [{{ entry.paper.arxiv_id }}](https://arxiv.org/abs/{{ entry.paper.arxiv_id }}) |
| **Authors** | {{ entry.paper.authors[:5] | join(', ') }}{% if entry.paper.authors | length > 5 %} *et al.*{% endif %} |
| **Published** | {{ entry.paper.published.strftime('%Y-%m-%d') }} |
{% if entry.paper.venue %}| **Venue** | {{ entry.paper.venue }} |
{% endif %}
{% if entry.paper.citations is not none %}| **Citations** | {{ entry.paper.citations }} |
{% endif %}
{% if entry.paper.code_url %}| **Code** | [{{ 'Official' if entry.paper.code_official else 'Community' }}]({{ entry.paper.code_url }}) |
{% endif %}
| **Quality** | {{ '%.2f' | format(entry.scores.quality) }} |

{% if entry.summary %}
### Summary

> {{ entry.summary.one_liner }}

{% if entry.summary.affiliations %}**Affiliations:** {{ entry.summary.affiliations }}

{% endif %}**Method:** {{ entry.summary.method }}

**Data & Benchmarks:** {{ entry.summary.data_benchmarks }}

**Key Results:** {{ entry.summary.key_results }}

**Novelty:** {{ entry.summary.novelty }}

**AD Relevance:** {{ entry.summary.ad_relevance }}

{% if entry.summary.limitations %}**Limitations:** {{ entry.summary.limitations }}{% endif %}

{% endif %}
{% if entry.paper.pdf_url %}[PDF]({{ entry.paper.pdf_url }}){% endif %}
{% if entry.paper.oa_pdf_url %} | [Open Access PDF]({{ entry.paper.oa_pdf_url }}){% endif %}

---

{% endfor %}

{% if digest.rejected %}
<details>
<summary>Reviewed but not included ({{ digest.rejected | length }} papers)</summary>

{% for result in digest.rejected %}
- **{{ result.paper.title }}** — {{ result.paper.authors[:3] | join(', ') }}{% if result.paper.authors | length > 3 %} et al.{% endif %}
  *{{ result.reason }}*
{% endfor %}

</details>
{% endif %}

*Generated by paperdigest v0.2.0*
```

Key changes:
- Scores line shows only `Quality` (no relevance/final)
- Added `Filtered out` count in header
- Added collapsible rejected papers section at bottom
- Version bumped to 0.2.0

**Step 2: Commit**

```bash
git add src/paperdigest/templates/digest.md.j2
git commit -m "feat: update digest template with rejected papers section, remove relevance score"
```

---

### Task 9: Update `config.yaml`

**Files:**
- Modify: `config.yaml`

**Step 1: Update config file**

```yaml
topic:
  name: "VLM/VLA for Autonomous Driving"
  description: >
    Papers about vision-language models (VLMs), vision-language-action models (VLAs),
    and multimodal foundation models applied to autonomous driving, including end-to-end
    driving, motion planning with language reasoning, and driving scene understanding.
    Also interested in relevant benchmarks and datasets for evaluating these systems.
  primary_keywords:
    - "autonomous driving"
    - "motion planning"
    - "driving"
  secondary_keywords:
    - "end-to-end"
    - "vision-language model"
    - "vlm"
    - "action"
    - "vla"
    - "reasoning"
  benchmarks:
    - "nuScenes"
    - "Waymo Open"
    - "CARLA"
    - "DriveLM"
    - "OpenScene"
    - "DriveVLM"
  arxiv_categories:
    - "cs.CV"
    - "cs.AI"
    - "cs.RO"
    - "cs.LG"

collection:
  lookback_days: 7
  max_results: 200
  blogs:
    enabled: true
    sources:
      - nvidia
      - waymo
  conferences:
    enabled: true
    years_back: 2
    venues:
      - CVPR
      - ICCV
      - ECCV
      - ICML
      - ICLR
      - AAAI
      - CoRL
      - ICRA
      - IROS
      - IV
      - ITSC

scoring:
  quality:
    w_venue: 0.25
    w_author: 0.20
    w_cite: 0.20
    w_code: 0.15
    w_fresh: 0.20
  venue_tiers:
    tier1:
      - "NeurIPS"
      - "ICML"
      - "ICLR"
      - "CVPR"
      - "ECCV"
      - "ICCV"
      - "AAAI"
      - "IJCAI"
      - "RSS"
      - "CoRL"
    tier2:
      - "IROS"
      - "ICRA"
      - "WACV"
      - "BMVC"
      - "ACCV"
      - "AISTATS"
      - "UAI"
    tier3:
      - "IV"
      - "ITSC"
      - "T-ITS"
      - "RA-L"
      - "T-RO"

digest:
  top_n: 20
  summarize_top_n: 10
  output_dir: "data/digests"

llm:
  filter:
    enabled: true
    model: "gpt-4o-mini"
    base_url: "https://api.openai.com/v1"
    temperature:
    max_completion_tokens: 256
    cost_control:
      max_cost_per_run: 0.10
      max_cost_per_month: 3.00
      input_cost_per_1k: 0.00015
      output_cost_per_1k: 0.0006
  summarizer:
    enabled: true
    model: "gpt-5-nano-2025-08-07"
    base_url: "https://api.openai.com/v1"
    temperature:
    max_completion_tokens: 16384
    max_text_chars: 50000
    cost_control:
      max_cost_per_run: 0.50
      max_cost_per_month: 10.00
      input_cost_per_1k: 0.00005
      output_cost_per_1k: 0.0004

delivery:
  markdown:
    enabled: true
  telegram:
    enabled: false

database:
  path: "data/papers.db"

pwc:
  links_path: "data/pwc_links.json"
```

**Step 2: Commit**

```bash
git add config.yaml
git commit -m "refactor: update config.yaml for split LLM config and topic description"
```

---

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

Update the CLAUDE.md to reflect:
- New pipeline flow
- New `filter` subcommand
- Split LLM config (`llm.filter` + `llm.summarizer`)
- `topic.description` field
- Removed `scoring.alpha`, `scoring.relevance`, `llm.use_full_text`
- New DB table `paper_filter_results`
- Updated `Scores` model (no `relevance`, has `llm_rank`)
- New `FilterResult` model
- New `filter.py` module
- Updated `Digest` model (has `rejected` field)

**Step 1: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for LLM filter/rank pipeline"
```

---

### Task 11: Run full test suite and fix any remaining issues

**Step 1: Run all tests**

Run: `python -m pytest tests/ -v`

**Step 2: Fix any failures**

Common issues to watch for:
- Old imports of `RelevanceWeights` in test files
- Tests referencing `scores.relevance` or `config.scoring.alpha`
- Tests referencing `config.llm.enabled` (now `config.llm.filter.enabled` / `config.llm.summarizer.enabled`)
- Summarizer tests using old `LLMConfig` constructor

**Step 3: Final commit**

```bash
git add -A
git commit -m "fix: resolve remaining test failures from pipeline refactor"
```

---

### Task 12: DB migration for existing databases

**Files:**
- Create: `src/paperdigest/migrate.py`

This is needed so existing `data/papers.db` files don't break. The `init_schema` uses `CREATE TABLE IF NOT EXISTS`, so new tables are fine, but the `scores` table has changed columns.

**Step 1: Create migration**

```python
"""Database migration for LLM filter/rank pipeline."""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


def migrate_scores_table(conn: sqlite3.Connection):
    """Migrate scores table: remove relevance, add llm_rank."""
    # Check if migration is needed
    cursor = conn.execute("PRAGMA table_info(scores)")
    columns = {row[1] for row in cursor.fetchall()}

    if "llm_rank" in columns:
        logger.info("scores table already migrated")
        return

    logger.info("Migrating scores table...")
    conn.executescript("""
        ALTER TABLE scores RENAME TO scores_old;

        CREATE TABLE scores (
            paper_id INTEGER PRIMARY KEY REFERENCES papers(id),
            quality REAL NOT NULL,
            llm_rank INTEGER NOT NULL DEFAULT 0,
            scored_at TEXT DEFAULT (datetime('now'))
        );

        INSERT INTO scores (paper_id, quality, llm_rank, scored_at)
        SELECT paper_id, quality, 0, scored_at FROM scores_old;

        DROP TABLE scores_old;

        CREATE INDEX IF NOT EXISTS idx_scores_llm_rank ON scores(llm_rank ASC);
    """)
    conn.commit()
    logger.info("scores table migration complete")
```

Add a call to this in `db.init_schema()`:
```python
def init_schema(self):
    self.conn.executescript(SCHEMA)
    self.conn.execute("PRAGMA foreign_keys=ON")
    self.conn.commit()
    # Run migrations for existing databases
    from .migrate import migrate_scores_table
    migrate_scores_table(self.conn)
```

**Step 2: Commit**

```bash
git add src/paperdigest/migrate.py src/paperdigest/db.py
git commit -m "feat: add DB migration for scores table schema change"
```
