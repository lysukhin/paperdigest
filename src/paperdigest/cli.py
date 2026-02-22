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


def cmd_digest(args):
    """Generate and deliver digest."""
    config = get_config(args)
    with Database(config.db_path) as db:
        db.init_schema()

        dry_run = getattr(args, "dry_run", False)

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


def cmd_run(args):
    """Full pipeline: fetch -> digest (includes filter, enrich, score, summarize, rank)."""
    logger.info("Starting full pipeline run...")
    cmd_fetch(args)
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

        if config.llm.filter.enabled or config.llm.summarizer.enabled:
            stats = db.get_llm_stats()
            monthly_budget = (
                config.llm.filter.cost_control.max_cost_per_month
                + config.llm.summarizer.cost_control.max_cost_per_month
            )
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

    # filter
    subparsers.add_parser("filter", parents=[common], help="Filter papers by LLM relevance")

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
        "filter": cmd_filter,
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
