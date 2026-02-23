"""SQLite database layer."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .models import Paper, Scores

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,           -- JSON list
    published TEXT NOT NULL,         -- ISO datetime
    updated TEXT,
    doi TEXT,
    categories TEXT,                 -- JSON list
    pdf_url TEXT,
    citations INTEGER,
    max_hindex INTEGER,
    venue TEXT,
    oa_pdf_url TEXT,
    code_url TEXT,
    code_official INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    digested_at TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id),
    quality REAL NOT NULL,
    llm_rank INTEGER NOT NULL DEFAULT 0,
    scored_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    paper_ids TEXT NOT NULL,         -- JSON list of paper IDs
    delivery_status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    run_id TEXT NOT NULL,
    papers_summarized INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    estimated_cost_usd REAL DEFAULT 0.0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS paper_filter_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL REFERENCES papers(id),
    run_date TEXT NOT NULL,
    relevant INTEGER NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_scores_final ON scores(llm_rank ASC, quality DESC);
CREATE INDEX IF NOT EXISTS idx_llm_usage_date ON llm_usage(date);
CREATE INDEX IF NOT EXISTS idx_filter_results_paper_date ON paper_filter_results(paper_id, run_date);
"""


class Database:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def init_schema(self):
        # Run migrations for existing databases before applying schema
        # (old scores table would cause index creation to fail on new columns)
        from .migrate import migrate_add_digested_at, migrate_scores_table
        migrate_scores_table(self.conn)
        migrate_add_digested_at(self.conn)

        self.conn.executescript(SCHEMA)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.commit()

    def close(self):
        self.conn.close()

    # --- Paper CRUD ---

    def upsert_paper(self, paper: Paper) -> int:
        """Insert or update a paper. Returns the db ID."""
        row = self.conn.execute(
            "SELECT id FROM papers WHERE arxiv_id = ?", (paper.arxiv_id,)
        ).fetchone()

        if row:
            self.conn.execute(
                """UPDATE papers SET
                    title=?, abstract=?, authors=?, published=?, updated=?,
                    doi=?, categories=?, pdf_url=?, citations=?, max_hindex=?,
                    venue=?, oa_pdf_url=?, code_url=?, code_official=?,
                    updated_at=datetime('now')
                WHERE arxiv_id=?""",
                (
                    paper.title,
                    paper.abstract,
                    json.dumps(paper.authors),
                    paper.published.isoformat(),
                    paper.updated.isoformat() if paper.updated else None,
                    paper.doi,
                    json.dumps(paper.categories),
                    paper.pdf_url,
                    paper.citations,
                    paper.max_hindex,
                    paper.venue,
                    paper.oa_pdf_url,
                    paper.code_url,
                    int(paper.code_official),
                    paper.arxiv_id,
                ),
            )
            self.conn.commit()
            return row["id"]
        else:
            cur = self.conn.execute(
                """INSERT INTO papers
                    (arxiv_id, title, abstract, authors, published, updated,
                     doi, categories, pdf_url, citations, max_hindex,
                     venue, oa_pdf_url, code_url, code_official)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paper.arxiv_id,
                    paper.title,
                    paper.abstract,
                    json.dumps(paper.authors),
                    paper.published.isoformat(),
                    paper.updated.isoformat() if paper.updated else None,
                    paper.doi,
                    json.dumps(paper.categories),
                    paper.pdf_url,
                    paper.citations,
                    paper.max_hindex,
                    paper.venue,
                    paper.oa_pdf_url,
                    paper.code_url,
                    int(paper.code_official),
                ),
            )
            self.conn.commit()
            return cur.lastrowid

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return self._row_to_paper(row) if row else None

    def get_paper_by_doi(self, doi: str) -> Paper | None:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE doi = ?", (doi,)
        ).fetchone()
        return self._row_to_paper(row) if row else None

    def get_all_papers(self) -> list[Paper]:
        rows = self.conn.execute("SELECT * FROM papers ORDER BY published DESC").fetchall()
        return [self._row_to_paper(r) for r in rows]

    def get_undigested_papers(self) -> list[Paper]:
        """Get papers not yet included in any digest."""
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE digested_at IS NULL ORDER BY published DESC"
        ).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def mark_papers_digested(self, paper_ids: list[int]):
        """Mark papers as included in a digest."""
        if not paper_ids:
            return
        placeholders = ",".join("?" * len(paper_ids))
        self.conn.execute(
            f"UPDATE papers SET digested_at = datetime('now') WHERE id IN ({placeholders})",
            paper_ids,
        )
        self.conn.commit()

    def get_unenriched_papers(self) -> list[Paper]:
        """Papers that haven't been enriched (no citations data)."""
        rows = self.conn.execute(
            "SELECT * FROM papers WHERE citations IS NULL ORDER BY published DESC"
        ).fetchall()
        return [self._row_to_paper(r) for r in rows]

    def update_enrichment(self, paper: Paper):
        """Update only enrichment fields for a paper."""
        self.conn.execute(
            """UPDATE papers SET
                citations=?, max_hindex=?, venue=?, oa_pdf_url=?,
                code_url=?, code_official=?, updated_at=datetime('now')
            WHERE arxiv_id=?""",
            (
                paper.citations,
                paper.max_hindex,
                paper.venue,
                paper.oa_pdf_url,
                paper.code_url,
                int(paper.code_official),
                paper.arxiv_id,
            ),
        )
        self.conn.commit()

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        return Paper(
            db_id=row["id"],
            arxiv_id=row["arxiv_id"],
            title=row["title"],
            abstract=row["abstract"],
            authors=json.loads(row["authors"]),
            published=datetime.fromisoformat(row["published"]),
            updated=datetime.fromisoformat(row["updated"]) if row["updated"] else None,
            doi=row["doi"],
            categories=json.loads(row["categories"]) if row["categories"] else [],
            pdf_url=row["pdf_url"],
            citations=row["citations"],
            max_hindex=row["max_hindex"],
            venue=row["venue"],
            oa_pdf_url=row["oa_pdf_url"],
            code_url=row["code_url"],
            code_official=bool(row["code_official"]),
        )

    # --- Scores ---

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
            scores = Scores(quality=row["quality"], llm_rank=row["llm_rank"])
            results.append((paper, scores))
        return results

    # --- Filter Results ---

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

    # --- Digests ---

    def log_digest(self, paper_ids: list[int], status: str = "delivered") -> int:
        cur = self.conn.execute(
            "INSERT INTO digests (date, paper_ids, delivery_status) VALUES (?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d"), json.dumps(paper_ids), status),
        )
        self.conn.commit()
        return cur.lastrowid

    # --- LLM Usage ---

    def log_llm_usage(
        self,
        run_id: str,
        papers_summarized: int,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
    ):
        self.conn.execute(
            """INSERT INTO llm_usage
                (date, run_id, papers_summarized, input_tokens, output_tokens, estimated_cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().strftime("%Y-%m-%d"),
                run_id,
                papers_summarized,
                input_tokens,
                output_tokens,
                estimated_cost,
            ),
        )
        self.conn.commit()

    def get_monthly_llm_cost(self, year: int | None = None, month: int | None = None) -> float:
        now = datetime.now()
        y = year or now.year
        m = month or now.month
        prefix = f"{y:04d}-{m:02d}"
        row = self.conn.execute(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) as total FROM llm_usage WHERE date LIKE ?",
            (f"{prefix}%",),
        ).fetchone()
        return row["total"]

    def get_llm_stats(self) -> dict:
        now = datetime.now()
        prefix = f"{now.year:04d}-{now.month:02d}"
        row = self.conn.execute(
            """SELECT
                COUNT(*) as runs,
                COALESCE(SUM(papers_summarized), 0) as total_papers,
                COALESCE(SUM(input_tokens), 0) as total_input,
                COALESCE(SUM(output_tokens), 0) as total_output,
                COALESCE(SUM(estimated_cost_usd), 0) as total_cost
            FROM llm_usage WHERE date LIKE ?""",
            (f"{prefix}%",),
        ).fetchone()
        runs = row["runs"]
        return {
            "month": prefix,
            "runs": runs,
            "total_papers_summarized": row["total_papers"],
            "total_input_tokens": row["total_input"],
            "total_output_tokens": row["total_output"],
            "total_cost_usd": row["total_cost"],
            "avg_cost_per_run": row["total_cost"] / runs if runs > 0 else 0,
            "avg_cost_per_paper": (
                row["total_cost"] / row["total_papers"]
                if row["total_papers"] > 0
                else 0
            ),
        }

    # --- Stats ---

    def get_paper_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as c FROM papers").fetchone()
        return row["c"]

    def get_scored_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as c FROM scores").fetchone()
        return row["c"]

    def get_digest_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as c FROM digests").fetchone()
        return row["c"]
