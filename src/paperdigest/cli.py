"""CLI entry point for paperdigest."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, load_config
from .db import Database
from .models import Digest, DigestEntry

logger = logging.getLogger("paperdigest")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_config(args) -> Config:
    config_path = getattr(args, "config", "config.yaml")
    return load_config(config_path)


def cmd_init(args):
    """Initialize database and download PWC links."""
    config = get_config(args)
    with Database(config.db_path) as db:
        db.init_schema()
        logger.info(f"Database initialized at {config.db_path}")

        if not args.skip_pwc:
            from .enrichment.pwc import download_pwc_links

            download_pwc_links(config.pwc_path)

    logger.info("Initialization complete")


def _get_blog_collectors(config):
    """Build blog collector instances based on config."""
    collectors = []
    blog_cfg = config.collection.blogs
    if not blog_cfg.enabled:
        return collectors

    source_map = {}
    if "nvidia" in blog_cfg.sources:
        from .collectors.nvidia import NvidiaCollector
        source_map["nvidia"] = NvidiaCollector
    if "waymo" in blog_cfg.sources:
        from .collectors.waymo import WaymoCollector
        source_map["waymo"] = WaymoCollector
    if "wayve" in blog_cfg.sources:
        from .collectors.wayve import WayveCollector
        source_map["wayve"] = WayveCollector

    for name in blog_cfg.sources:
        cls = source_map.get(name)
        if cls:
            collectors.append(cls(config))
        else:
            logger.warning(f"Unknown blog source: {name}")
    return collectors


def cmd_fetch(args):
    """Collect papers from arXiv and configured blog sources."""
    config = get_config(args)
    with Database(config.db_path) as db:
        db.init_schema()

        from .collectors.arxiv import ArxivCollector
        from .dedup import dedup_papers

        # arXiv
        collector = ArxivCollector(config)
        papers = collector.collect()

        # Blog sources
        for blog_collector in _get_blog_collectors(config):
            try:
                blog_papers = blog_collector.collect()
                papers.extend(blog_papers)
            except Exception:
                logger.exception(f"Failed to collect from {blog_collector.source_name}")

        # Conference proceedings (DBLP)
        if config.collection.conferences.enabled:
            from .collectors.dblp import DBLPCollector

            try:
                dblp = DBLPCollector(config)
                papers.extend(dblp.collect())
            except Exception:
                logger.exception("Failed to collect from DBLP")

        new_papers = dedup_papers(papers, db)

        for paper in new_papers:
            db.upsert_paper(paper)

        logger.info(f"Stored {len(new_papers)} new papers (of {len(papers)} collected)")


def cmd_enrich(args):
    """Enrich stored papers with external data."""
    config = get_config(args)
    with Database(config.db_path) as db:
        db.init_schema()

        papers = db.get_unenriched_papers()
        if not papers:
            logger.info("No papers to enrich")
            return

        logger.info(f"Enriching {len(papers)} papers...")

        from .enrichment.pwc import enrich_with_pwc
        from .enrichment.semantic_scholar import enrich_papers

        papers = enrich_papers(papers, config)
        papers = enrich_with_pwc(papers, config)

        for paper in papers:
            db.update_enrichment(paper)

        logger.info("Enrichment complete")


def cmd_score(args):
    """Score all papers."""
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

        logger.info(f"Scored {len(scored)} papers")


def cmd_digest(args):
    """Generate and deliver digest."""
    config = get_config(args)
    with Database(config.db_path) as db:
        db.init_schema()

        dry_run = getattr(args, "dry_run", False)

        top = db.get_top_scored_papers(limit=config.digest.top_n)
        if not top:
            logger.info("No scored papers. Run 'score' first.")
            return

        # Optional LLM summarization
        summaries = {}
        if config.llm.enabled and not dry_run:
            from .summarizer import Summarizer

            summarizer = Summarizer(config, db)
            summarize_papers = [p for p, _ in top[: config.digest.summarize_top_n]]
            summaries = summarizer.summarize_papers(summarize_papers)
        elif dry_run:
            logger.info("Dry run — skipping LLM summarization")

        # Build digest
        entries = []
        for rank, (paper, scores) in enumerate(top, 1):
            summary = summaries.get(paper.arxiv_id)
            entries.append(DigestEntry(paper=paper, scores=scores, rank=rank, summary=summary))

        total_papers = db.get_paper_count()
        digest = Digest(
            date=datetime.now(timezone.utc),
            topic_name=config.topic.name,
            entries=entries,
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

        # Log digest
        paper_ids = [p.db_id for p, _ in top]
        status = "delivered" if tg_ok else "partial_delivery"
        db.log_digest(paper_ids, status=status)


def cmd_run(args):
    """Full pipeline: fetch -> enrich -> score -> digest."""
    logger.info("Starting full pipeline run...")
    cmd_fetch(args)
    cmd_enrich(args)
    cmd_score(args)
    cmd_digest(args)
    logger.info("Pipeline run complete")


def cmd_serve(args):
    """Start the web dashboard."""
    config = get_config(args)

    from .web import run_server

    run_server(config)


def cmd_stats(args):
    """Show database statistics."""
    config = get_config(args)
    with Database(config.db_path) as db:
        db.init_schema()

        paper_count = db.get_paper_count()
        scored_count = db.get_scored_count()
        digest_count = db.get_digest_count()

        print(f"Database: {config.db_path}")
        print(f"Papers:   {paper_count}")
        print(f"Scored:   {scored_count}")
        print(f"Digests:  {digest_count}")

        if config.llm.enabled:
            stats = db.get_llm_stats()
            monthly_budget = config.llm.cost_control.max_cost_per_month
            remaining = monthly_budget - stats["total_cost_usd"]
            print(f"\nLLM Usage ({stats['month']}):")
            print(f"  Runs:                {stats['runs']}")
            print(f"  Papers summarized:   {stats['total_papers_summarized']}")
            print(f"  Total tokens:        {stats['total_input_tokens']} in / {stats['total_output_tokens']} out")
            print(f"  Month cost:          ${stats['total_cost_usd']:.4f}")
            print(f"  Budget remaining:    ${remaining:.2f} / ${monthly_budget:.2f}")
            print(f"  Avg cost/run:        ${stats['avg_cost_per_run']:.4f}")
            print(f"  Avg cost/paper:      ${stats['avg_cost_per_paper']:.4f}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="paperdigest",
        description="Automated research paper digest system",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # Common parent for --config
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    subparsers.add_parser("run", parents=[common], help="Full pipeline: fetch -> enrich -> score -> digest")

    # fetch
    subparsers.add_parser("fetch", parents=[common], help="Collect papers from arXiv")

    # enrich
    subparsers.add_parser("enrich", parents=[common], help="Enrich stored papers")

    # score
    subparsers.add_parser("score", parents=[common], help="Score all papers")

    # digest
    digest_parser = subparsers.add_parser("digest", parents=[common], help="Generate and deliver digest")
    digest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls, produce digest without summaries",
    )

    # init
    init_parser = subparsers.add_parser("init", parents=[common], help="Initialize DB and download PWC data")
    init_parser.add_argument(
        "--skip-pwc",
        action="store_true",
        help="Skip downloading PWC links",
    )

    # serve
    subparsers.add_parser("serve", parents=[common], help="Start web dashboard")

    # stats
    subparsers.add_parser("stats", parents=[common], help="Show database statistics")

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "run": cmd_run,
        "fetch": cmd_fetch,
        "enrich": cmd_enrich,
        "score": cmd_score,
        "digest": cmd_digest,
        "init": cmd_init,
        "serve": cmd_serve,
        "stats": cmd_stats,
    }

    try:
        commands[args.command](args)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
