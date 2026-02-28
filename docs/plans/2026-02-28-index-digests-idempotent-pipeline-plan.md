# Index-Based Digests + Idempotent Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace date-based digest naming with auto-incrementing index numbers, and cache LLM summaries so expensive operations run at most once per paper.

**Architecture:** Add `digest_number` column to `digests` table (auto-increment via `MAX+1`), add `paper_summaries` table for LLM summary caching. Reorder delivery flow so `log_digest` runs before file writes. Summarizer checks DB before calling LLM.

**Tech Stack:** Python, SQLite, FastAPI, Jinja2, pytest

---

### Task 1: Create Branch

**Step 1: Create and switch to feature branch**

```bash
git checkout -b feature/index-digests-idempotent-pipeline
```

**Step 2: Verify**

```bash
git branch --show-current
```
Expected: `feature/index-digests-idempotent-pipeline`

---

### Task 2: DB Schema — Add `paper_summaries` Table and Summary Methods

**Files:**
- Modify: `src/paperdigest/db.py` (SCHEMA string + new methods)
- Test: `tests/test_db.py`

**Step 1: Write failing tests for summary persistence**

Add to `tests/test_db.py`:

```python
from paperdigest.models import Paper, Scores, Summary

class TestSummaryPersistence:
    def test_upsert_and_get_summary(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        summary = Summary(
            one_liner="A novel approach",
            affiliations="MIT",
            method="We propose X",
            data_benchmarks="nuScenes",
            key_results="95% accuracy",
            novelty="First to do Y",
            ad_relevance="Driving perception",
            limitations="Simulation only",
        )
        db.upsert_summary(paper_id, summary)
        result = db.get_summary(paper_id)
        assert result is not None
        assert result.one_liner == "A novel approach"
        assert result.affiliations == "MIT"
        assert result.method == "We propose X"
        assert result.data_benchmarks == "nuScenes"
        assert result.key_results == "95% accuracy"
        assert result.novelty == "First to do Y"
        assert result.ad_relevance == "Driving perception"
        assert result.limitations == "Simulation only"
        db.close()

    def test_get_summary_returns_none_when_missing(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        result = db.get_summary(paper_id)
        assert result is None
        db.close()

    def test_upsert_summary_overwrites(self, tmp_path):
        db = _setup_db(tmp_path)
        paper_id = db.upsert_paper(_make_paper())
        db.upsert_summary(paper_id, Summary(one_liner="Old"))
        db.upsert_summary(paper_id, Summary(one_liner="New"))
        result = db.get_summary(paper_id)
        assert result.one_liner == "New"
        db.close()
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_db.py::TestSummaryPersistence -v
```
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'upsert_summary'`

**Step 3: Add `paper_summaries` table to SCHEMA and implement methods**

In `src/paperdigest/db.py`, add to the `SCHEMA` string (after the `paper_filter_results` table):

```sql
CREATE TABLE IF NOT EXISTS paper_summaries (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id),
    one_liner TEXT NOT NULL DEFAULT '',
    affiliations TEXT NOT NULL DEFAULT '',
    method TEXT NOT NULL DEFAULT '',
    data_benchmarks TEXT NOT NULL DEFAULT '',
    key_results TEXT NOT NULL DEFAULT '',
    novelty TEXT NOT NULL DEFAULT '',
    ad_relevance TEXT NOT NULL DEFAULT '',
    limitations TEXT NOT NULL DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);
```

Add import at top of `db.py`:
```python
from .models import Paper, Scores, Summary
```

Add methods to `Database` class (after the `get_rejected_papers` method, before `# --- Digests ---`):

```python
# --- Summaries ---

def get_summary(self, paper_id: int) -> Summary | None:
    """Get cached LLM summary for a paper."""
    row = self.conn.execute(
        "SELECT * FROM paper_summaries WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    if not row:
        return None
    return Summary(
        one_liner=row["one_liner"],
        affiliations=row["affiliations"],
        method=row["method"],
        data_benchmarks=row["data_benchmarks"],
        key_results=row["key_results"],
        novelty=row["novelty"],
        ad_relevance=row["ad_relevance"],
        limitations=row["limitations"],
    )

def upsert_summary(self, paper_id: int, summary: Summary):
    """Store or update an LLM summary for a paper."""
    self.conn.execute(
        """INSERT INTO paper_summaries
            (paper_id, one_liner, affiliations, method, data_benchmarks,
             key_results, novelty, ad_relevance, limitations)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            one_liner=excluded.one_liner,
            affiliations=excluded.affiliations,
            method=excluded.method,
            data_benchmarks=excluded.data_benchmarks,
            key_results=excluded.key_results,
            novelty=excluded.novelty,
            ad_relevance=excluded.ad_relevance,
            limitations=excluded.limitations,
            created_at=datetime('now')""",
        (
            paper_id,
            summary.one_liner,
            summary.affiliations,
            summary.method,
            summary.data_benchmarks,
            summary.key_results,
            summary.novelty,
            summary.ad_relevance,
            summary.limitations,
        ),
    )
    self.conn.commit()
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_db.py::TestSummaryPersistence -v
```
Expected: 3 PASSED

**Step 5: Run full test suite to check nothing broke**

```bash
python -m pytest tests/ -v
```
Expected: All existing tests PASS

**Step 6: Commit**

```bash
git add src/paperdigest/db.py tests/test_db.py
git commit -m "feat: add paper_summaries table for LLM summary caching"
```

---

### Task 3: Summarizer — Use Cached Summaries

**Files:**
- Modify: `src/paperdigest/summarizer.py`
- Test: `tests/test_summarizer.py`

**Step 1: Write failing test for summary caching**

Add to `tests/test_summarizer.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_summarizer.py::TestSummaryCaching -v
```
Expected: FAIL — first test fails because summarizer doesn't check DB cache yet

**Step 3: Modify `summarize_paper()` in `src/paperdigest/summarizer.py`**

At the beginning of `summarize_paper()`, before the budget check, add DB cache lookup:

```python
def summarize_paper(self, paper: Paper) -> Summary | None:
    """Summarize a single paper. Returns None on failure."""
    # Check DB cache first
    if paper.db_id is not None:
        cached = self.db.get_summary(paper.db_id)
        if cached is not None:
            logger.info(f"Using cached summary for {paper.arxiv_id}")
            return cached

    ok, reason = self._check_budget()
    # ... rest of existing method unchanged ...
```

At the end of `summarize_paper()`, after the `self.run_papers += 1` line and before the `return Summary(...)`, persist the summary:

```python
    self.run_papers += 1

    summary = Summary(
        one_liner=data.get("one_liner", ""),
        affiliations=data.get("affiliations", ""),
        method=data.get("method", ""),
        data_benchmarks=data.get("data_benchmarks", ""),
        key_results=data.get("key_results", ""),
        novelty=data.get("novelty", ""),
        ad_relevance=data.get("ad_relevance", ""),
        limitations=data.get("limitations", ""),
    )

    # Cache to DB
    if paper.db_id is not None:
        self.db.upsert_summary(paper.db_id, summary)

    return summary
```

(Replace the existing inline `return Summary(...)` construction.)

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_summarizer.py -v
```
Expected: All PASS (including new TestSummaryCaching tests)

**Step 5: Commit**

```bash
git add src/paperdigest/summarizer.py tests/test_summarizer.py
git commit -m "feat: cache LLM summaries in DB, skip re-summarization"
```

---

### Task 4: DB Schema — Add `digest_number` Column and Migration

**Files:**
- Modify: `src/paperdigest/db.py` (SCHEMA + `log_digest` method)
- Modify: `src/paperdigest/migrate.py`
- Test: `tests/test_db.py`

**Step 1: Write failing tests for digest numbering**

Add to `tests/test_db.py`:

```python
class TestDigestNumbering:
    def test_log_digest_returns_digest_number(self, tmp_path):
        db = _setup_db(tmp_path)
        p_id = db.upsert_paper(_make_paper())
        number = db.log_digest([p_id], status="delivered")
        assert number == 1

    def test_digest_numbers_auto_increment(self, tmp_path):
        db = _setup_db(tmp_path)
        p1 = db.upsert_paper(_make_paper(arxiv_id="2401.00001"))
        p2 = db.upsert_paper(_make_paper(arxiv_id="2401.00002"))
        n1 = db.log_digest([p1], status="delivered")
        n2 = db.log_digest([p2], status="delivered")
        assert n1 == 1
        assert n2 == 2
        db.close()

    def test_update_digest_status(self, tmp_path):
        db = _setup_db(tmp_path)
        p_id = db.upsert_paper(_make_paper())
        number = db.log_digest([p_id], status="pending")
        db.update_digest_status(number, "delivered")
        row = db.conn.execute(
            "SELECT delivery_status FROM digests WHERE digest_number = ?", (number,)
        ).fetchone()
        assert row["delivery_status"] == "delivered"
        db.close()

    def test_get_next_digest_number_with_no_digests(self, tmp_path):
        db = _setup_db(tmp_path)
        p_id = db.upsert_paper(_make_paper())
        number = db.log_digest([p_id])
        assert number == 1
        db.close()
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_db.py::TestDigestNumbering -v
```
Expected: FAIL

**Step 3: Add migration for `digest_number`**

In `src/paperdigest/migrate.py`, add:

```python
def migrate_add_digest_number(conn: sqlite3.Connection):
    """Add digest_number column to digests table."""
    cursor = conn.execute("PRAGMA table_info(digests)")
    columns = {row[1] for row in cursor.fetchall()}
    if "digest_number" in columns:
        logger.debug("digests table already has digest_number column")
        return
    if "date" not in columns:
        # Table doesn't exist yet — schema creation will handle it
        return
    logger.info("Adding digest_number column to digests table...")
    conn.execute("ALTER TABLE digests ADD COLUMN digest_number INTEGER")
    # Backfill existing rows with sequential numbers
    rows = conn.execute("SELECT id FROM digests ORDER BY created_at ASC, id ASC").fetchall()
    for i, row in enumerate(rows, start=1):
        conn.execute("UPDATE digests SET digest_number = ? WHERE id = ?", (i, row[0]))
    if rows:
        logger.info(f"Backfilled digest_number for {len(rows)} existing digests")
    conn.commit()
    logger.info("digest_number migration complete")
```

**Step 4: Update SCHEMA in `src/paperdigest/db.py`**

Change the `digests` table in SCHEMA to include `digest_number`:

```sql
CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_number INTEGER NOT NULL,
    date TEXT NOT NULL,
    paper_ids TEXT NOT NULL,
    delivery_status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT (datetime('now'))
);
```

**Step 5: Call migration in `init_schema()`**

In `db.py` `init_schema()`, add the new migration call:

```python
def init_schema(self):
    from .migrate import migrate_add_digested_at, migrate_scores_table, migrate_add_digest_number
    migrate_scores_table(self.conn)
    migrate_add_digested_at(self.conn)
    migrate_add_digest_number(self.conn)

    self.conn.executescript(SCHEMA)
    self.conn.execute("PRAGMA foreign_keys=ON")
    self.conn.commit()
```

**Step 6: Update `log_digest()` to auto-assign digest_number and return it**

Replace the existing `log_digest` method:

```python
def log_digest(self, paper_ids: list[int], status: str = "delivered") -> int:
    """Log a digest and return its digest number."""
    row = self.conn.execute("SELECT COALESCE(MAX(digest_number), 0) FROM digests").fetchone()
    next_number = row[0] + 1
    self.conn.execute(
        "INSERT INTO digests (digest_number, date, paper_ids, delivery_status) VALUES (?, ?, ?, ?)",
        (next_number, datetime.now().strftime("%Y-%m-%d"), json.dumps(paper_ids), status),
    )
    self.conn.commit()
    return next_number
```

**Step 7: Add `update_digest_status()` method**

```python
def update_digest_status(self, digest_number: int, status: str):
    """Update delivery status for a digest."""
    self.conn.execute(
        "UPDATE digests SET delivery_status = ? WHERE digest_number = ?",
        (status, digest_number),
    )
    self.conn.commit()
```

**Step 8: Run tests to verify they pass**

```bash
python -m pytest tests/test_db.py -v
```
Expected: All PASS

**Step 9: Add migration test**

Add to `tests/test_db.py` inside the `TestMigration` class:

```python
def test_migrate_backfills_digest_number(self, tmp_path):
    """Migration assigns sequential digest_number to existing digests."""
    db_path = tmp_path / "digest_num.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Create old schema without digest_number
    conn.executescript("""
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            authors TEXT NOT NULL,
            published TEXT NOT NULL,
            updated TEXT, doi TEXT, categories TEXT, pdf_url TEXT,
            citations INTEGER, max_hindex INTEGER, venue TEXT,
            oa_pdf_url TEXT, code_url TEXT, code_official INTEGER DEFAULT 0,
            created_at TEXT, updated_at TEXT, digested_at TEXT
        );
        CREATE TABLE scores (
            paper_id INTEGER PRIMARY KEY REFERENCES papers(id),
            quality REAL NOT NULL,
            llm_rank INTEGER NOT NULL DEFAULT 0,
            scored_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE digests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            paper_ids TEXT NOT NULL,
            delivery_status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.execute("INSERT INTO digests (date, paper_ids) VALUES ('2024-01-01', '[1]')")
    conn.execute("INSERT INTO digests (date, paper_ids) VALUES ('2024-01-02', '[2]')")
    conn.commit()
    conn.close()

    with Database(db_path) as db:
        db.init_schema()
        rows = db.conn.execute(
            "SELECT digest_number FROM digests ORDER BY digest_number"
        ).fetchall()
        assert [r[0] for r in rows] == [1, 2]
```

**Step 10: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: All PASS

**Step 11: Commit**

```bash
git add src/paperdigest/db.py src/paperdigest/migrate.py tests/test_db.py
git commit -m "feat: add digest_number column with auto-increment and migration"
```

---

### Task 5: Model — Add `number` to `Digest`

**Files:**
- Modify: `src/paperdigest/models.py`

**Step 1: Add `number` field to `Digest` dataclass**

In `src/paperdigest/models.py`, change the `Digest` class:

```python
@dataclass
class Digest:
    """A complete digest ready for delivery."""

    date: datetime
    topic_name: str
    number: int = 0
    entries: list[DigestEntry] = field(default_factory=list)
    rejected: list[FilterResult] = field(default_factory=list)
    total_collected: int = 0
    total_new: int = 0
```

**Step 2: Run tests to ensure nothing breaks**

```bash
python -m pytest tests/ -v
```
Expected: All PASS (default `number=0` is backward-compatible)

**Step 3: Commit**

```bash
git add src/paperdigest/models.py
git commit -m "feat: add number field to Digest model"
```

---

### Task 6: Markdown Delivery — Use Digest Number for Filename

**Files:**
- Modify: `src/paperdigest/delivery/markdown.py`

**Step 1: Update `deliver_markdown` to use `digest.number`**

Replace the filename line in `deliver_markdown()`:

```python
def deliver_markdown(digest: Digest, config: Config) -> Path:
    """Write digest to a markdown file."""
    output_dir = config.digest_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"digest_{digest.number:03d}.md"
    output_path = output_dir / filename

    content = render_digest(digest, config)
    output_path.write_text(content)

    logger.info(f"Digest written to {output_path}")
    return output_path
```

**Step 2: Run tests**

```bash
python -m pytest tests/ -v
```
Expected: All PASS

**Step 3: Commit**

```bash
git add src/paperdigest/delivery/markdown.py
git commit -m "feat: use digest number for markdown filename"
```

---

### Task 7: Telegram Delivery — Use Digest Number in URL

**Files:**
- Modify: `src/paperdigest/delivery/telegram.py`
- Modify: `tests/test_telegram.py`

**Step 1: Update Telegram URL construction**

In `src/paperdigest/delivery/telegram.py`, change `deliver_telegram()` (around line 96-99):

```python
# Replace:
#   date_str = digest.date.strftime("%Y-%m-%d")
#   digest_url = f"{public_url}/digest/{date_str}"
# With:
    if public_url:
        digest_url = f"{public_url}/digest/{digest.number}"
```

(Delete the `date_str` line and update the `digest_url` line.)

**Step 2: Update test assertion**

In `tests/test_telegram.py`, the `_make_digest` helper creates a `Digest` — add `number` parameter:

```python
def _make_digest(n_entries: int = 5, with_summaries: bool = True, number: int = 42) -> Digest:
    """Create a digest with N entries."""
    entries = []
    for i in range(1, n_entries + 1):
        one_liner = f"Summary for paper {i}" if with_summaries else ""
        entries.append(_make_entry(
            rank=i,
            title=f"Paper Title Number {i}",
            one_liner=one_liner,
            arxiv_id=f"2401.{i:05d}",
        ))
    return Digest(
        date=datetime(2026, 2, 22, tzinfo=timezone.utc),
        topic_name="VLM/VLA for AD",
        number=number,
        entries=entries,
        total_collected=45,
        total_new=12,
    )
```

Update the inline button URL assertion in `test_sends_with_inline_button_when_public_url_set`:

```python
# Change:
#   assert markup["inline_keyboard"][0][0]["url"] == "https://digest.example.com/digest/2026-02-22"
# To:
assert markup["inline_keyboard"][0][0]["url"] == "https://digest.example.com/digest/42"
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_telegram.py -v
```
Expected: All PASS

**Step 4: Commit**

```bash
git add src/paperdigest/delivery/telegram.py tests/test_telegram.py
git commit -m "feat: use digest number in Telegram URL"
```

---

### Task 8: Web Dashboard — Use Digest Number in Routes

**Files:**
- Modify: `src/paperdigest/web.py`
- Modify: `src/paperdigest/templates/web/index.html`
- Modify: `src/paperdigest/templates/web/digest_view.html`

**Step 1: Update web route from `{date}` to `{number}`**

In `src/paperdigest/web.py`:

Change the route handler:

```python
@app.get("/digest/{number:int}", response_class=HTMLResponse)
async def view_digest(request: Request, number: int):
    if number < 1:
        return HTMLResponse("Invalid digest number", status_code=400)
    content = _render_digest(config, number)
    if content is None:
        return HTMLResponse("Digest not found", status_code=404)
    return templates.TemplateResponse(
        "digest_view.html",
        {
            "request": request,
            "content": content,
            "number": number,
            "topic": config.topic.name,
        },
    )
```

Update `_list_digests()`:

```python
def _list_digests(config: Config) -> list[dict]:
    digest_dir = config.digest_dir
    if not digest_dir.exists():
        return []

    digests = []
    for path in sorted(digest_dir.glob("digest_*.md"), reverse=True):
        meta = _parse_digest_meta(path)
        if meta:
            digests.append(meta)
    return digests
```

Update `_parse_digest_meta()`:

```python
def _parse_digest_meta(path: Path) -> dict | None:
    match = re.search(r"digest_(\d+)", path.name)
    if not match:
        return None

    number = int(match.group(1))
    content = path.read_text()

    topic = ""
    papers_count = 0
    first_papers: list[str] = []
    date_str = ""

    for line in content.split("\n"):
        if line.startswith("# Paper Digest:"):
            topic = line.replace("# Paper Digest:", "").strip()
        elif "Ranked:" in line:
            m = re.search(r"Ranked:\*\*\s*(\d+)", line)
            if m:
                papers_count = int(m.group(1))
        elif line.startswith("## ") and len(first_papers) < 3:
            title = re.sub(r"^##\s+\d+\.\s*", "", line).strip()
            if title:
                first_papers.append(title)

    return {
        "number": number,
        "topic": topic,
        "papers_count": papers_count,
        "preview_titles": first_papers,
    }
```

Update `_render_digest()`:

```python
def _render_digest(config: Config, number: int) -> str | None:
    import markdown

    path = config.digest_dir / f"digest_{number:03d}.md"
    if not path.exists():
        return None

    md = markdown.Markdown(extensions=["tables", "fenced_code"])
    text = path.read_text()
    text = text.replace("<details>", '<details markdown="1">')
    md = markdown.Markdown(extensions=["tables", "fenced_code", "md_in_html"])
    return md.convert(text)
```

**Step 2: Update `index.html` template**

In `src/paperdigest/templates/web/index.html`, change the digest link (line 180):

```html
<a href="/digest/{{ d.number }}" class="digest-item">
    <div>
        <div class="digest-date">#{{ d.number }}</div>
```

**Step 3: Update `digest_view.html` template**

In `src/paperdigest/templates/web/digest_view.html`, change the title block (line 2):

```html
{% block title %}Digest #{{ number }}{% endblock %}
```

**Step 4: Run tests**

```bash
python -m pytest tests/ -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add src/paperdigest/web.py src/paperdigest/templates/web/index.html src/paperdigest/templates/web/digest_view.html
git commit -m "feat: use digest number in web routes and templates"
```

---

### Task 9: CLI Orchestration — Reorder Delivery to Use Digest Number

**Files:**
- Modify: `src/paperdigest/cli.py`

This is the critical change. Currently `_cmd_digest_inner()` calls `deliver_markdown` then `log_digest`. We need to:
1. Call `log_digest` first (to get the number)
2. Set `digest.number`
3. Call `deliver_markdown` and `deliver_telegram`
4. Update digest status

**Step 1: Rewrite the deliver section of `_cmd_digest_inner()`**

In `src/paperdigest/cli.py`, replace the delivery block (lines 256-271) with:

```python
    # Deliver
    from .delivery.markdown import deliver_markdown

    with tracker.stage("Deliver") as stage:
        # Log digest first to get the number
        paper_ids = [p.db_id for p in top_papers]
        digest_number = db.log_digest(paper_ids, status="pending")
        digest.number = digest_number

        md_path = deliver_markdown(digest, config)
        stage.set_detail(str(md_path))

        tg_ok = True
        if config.delivery.telegram.enabled:
            from .delivery.telegram import deliver_telegram
            tg_ok = deliver_telegram(digest, config)

        status = "delivered" if tg_ok else "partial_delivery"
        db.update_digest_status(digest_number, status)
        db.mark_papers_digested(paper_ids)
```

**Step 2: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: All PASS

**Step 3: Commit**

```bash
git add src/paperdigest/cli.py
git commit -m "feat: reorder delivery to use digest number from DB"
```

---

### Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update relevant sections**

Update the Architecture / Key Design Decisions section to reflect:
- Digest files use index-based naming (`digest_001.md`)
- `log_digest()` returns digest number (auto-increment)
- LLM summaries are cached in `paper_summaries` table
- Web routes use `/digest/{number}`
- Telegram links use digest number

Update the Database section to add:
- `paper_summaries` — cached LLM summaries, one row per paper

Update the Pipeline Flow description to note:
- Summarizer checks DB cache before calling LLM

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for index-based digests and summary caching"
```

---

### Task 11: Final Verification

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: All PASS

**Step 2: Verify the migration works on a fresh DB**

```bash
python -m paperdigest init --skip-pwc --config config.yaml
```
Expected: DB created successfully with new schema
