"""Database migration for LLM filter/rank pipeline."""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


def migrate_scores_table(conn: sqlite3.Connection):
    """Migrate scores table: remove relevance/final, add llm_rank."""
    cursor = conn.execute("PRAGMA table_info(scores)")
    columns = {row[1] for row in cursor.fetchall()}

    if "llm_rank" in columns:
        logger.debug("scores table already migrated")
        return

    if "relevance" not in columns:
        # Table doesn't exist yet or already has new schema
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
    """)
    conn.commit()
    logger.info("scores table migration complete")
