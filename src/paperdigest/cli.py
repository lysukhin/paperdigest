"""CLI entry point for paperdigest."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, load_config
from .db import Database
from .models import Digest, DigestEntry
from .progress import NullTracker, PipelineTracker, create_tracker

logger = logging.getLogger("paperdigest")


def _is_interactive() -> bool:
    if os.environ.get("NO_PROGRESS"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


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
    interactive = _is_interactive()
    tracker = create_tracker(interactive)
    with Database(config.db_path) as db, tracker:
        with tracker.stage("Init DB") as stage:
            db.init_schema()
            stage.set_detail(str(config.db_path))

        if not args.skip_pwc:
            from .enrichment.pwc import download_pwc_links

            with tracker.stage("PWC Links") as stage:
                download_pwc_links(config.pwc_path)
                stage.set_detail(str(config.pwc_path))
        else:
            tracker.skip_stage("PWC Links")

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


def _cmd_fetch_inner(config, db, tracker):
    """Core fetch logic, shared between cmd_fetch and cmd_run."""
    from .collectors.arxiv import ArxivCollector
    from .dedup import dedup_papers

    with tracker.stage("Fetch") as stage:
        collector = ArxivCollector(config)
        papers = collector.collect()
        stage.set_detail(f"{len(papers)} arXiv")

        for blog_collector in _get_blog_collectors(config):
            try:
                blog_papers = blog_collector.collect()
                papers.extend(blog_papers)
            except Exception:
                logger.exception(f"Failed to collect from {blog_collector.source_name}")

        if config.collection.conferences.enabled:
            from .collectors.dblp import DBLPCollector

            try:
                dblp = DBLPCollector(config)
                papers.extend(dblp.collect())
            except Exception:
                logger.exception("Failed to collect from DBLP")

        stage.set_detail(f"{len(papers)} papers collected")

    with tracker.stage("Dedup") as stage:
        new_papers = dedup_papers(papers, db)
        stage.set_detail(f"{len(papers)} \u2192 {len(new_papers)} new")

    for paper in new_papers:
        db.upsert_paper(paper)

    logger.info(f"Stored {len(new_papers)} new papers (of {len(papers)} collected)")


def _cmd_digest_inner(config, db, tracker, dry_run=False):
    """Core digest logic, shared between cmd_digest and cmd_run."""
    papers = db.get_all_papers()
    if not papers:
        logger.info("No papers found. Run 'fetch' first.")
        return

    # LLM filter
    rejected_results = []
    if config.llm.filter.enabled and not dry_run:
        from .filter import PaperFilter

        with tracker.stage("Filter", total=len(papers)) as stage:
            filt = PaperFilter(config, db)
            papers, rejected_results = filt.filter_papers(papers, progress=stage)
            stage.set_detail(f"{len(papers)} relevant, {len(rejected_results)} rejected")
            stage.set_cost(filt.run_cost)
        if not papers:
            logger.info("No relevant papers after filtering")
            return
    else:
        tracker.skip_stage("Filter")

    # Enrich survivors (if not already enriched)
    from .enrichment.pwc import enrich_with_pwc
    from .enrichment.semantic_scholar import enrich_papers

    unenriched = [p for p in papers if p.citations is None]
    if unenriched:
        with tracker.stage("Enrich", total=len(unenriched)) as stage:
            unenriched = enrich_papers(unenriched, config, progress=stage)
            unenriched = enrich_with_pwc(unenriched, config)
            for paper in unenriched:
                db.update_enrichment(paper)
            papers = [db.get_paper_by_arxiv_id(p.arxiv_id) for p in papers]
            papers = [p for p in papers if p is not None]
            stage.set_detail(f"{len(unenriched)} enriched")
    else:
        tracker.skip_stage("Enrich")

    # Quality scoring
    from .scoring import score_papers as compute_quality_scores

    with tracker.stage("Score") as stage:
        scored = compute_quality_scores(papers, config)
        for paper, scores in scored:
            db.upsert_scores(paper.db_id, scores)
        stage.set_detail(f"{len(scored)} papers scored")

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

        with tracker.stage("Summarize", total=len(summarize_subset)) as stage:
            summaries = summarizer.summarize_papers(summarize_subset, progress=stage)
            stage.set_detail(f"{len(summaries)} summarized")
            stage.set_cost(summarizer.run_cost)

        with tracker.stage("Rank") as stage:
            ranking = summarizer.rank_papers(top_papers, summaries, quality_map)
            stage.set_detail(f"{len(ranking)} ranked")
    else:
        if dry_run:
            logger.info("Dry run \u2014 skipping LLM summarization and ranking")
        tracker.skip_stage("Summarize")
        tracker.skip_stage("Rank")

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

    with tracker.stage("Deliver") as stage:
        md_path = deliver_markdown(digest, config)
        stage.set_detail(str(md_path))

        tg_ok = True
        if config.delivery.telegram.enabled:
            from .delivery.telegram import deliver_telegram
            tg_ok = deliver_telegram(digest, config)

        paper_ids = [p.db_id for p in top_papers]
        status = "delivered" if tg_ok else "partial_delivery"
        db.log_digest(paper_ids, status=status)

    logger.info(f"Markdown digest: {md_path}")


def cmd_fetch(args):
    """Collect papers from arXiv and configured blog sources."""
    config = get_config(args)
    interactive = _is_interactive()
    tracker = create_tracker(interactive)
    with Database(config.db_path) as db:
        db.init_schema()
        with tracker:
            _cmd_fetch_inner(config, db, tracker)


def cmd_enrich(args):
    """Enrich stored papers with external data."""
    config = get_config(args)
    interactive = _is_interactive()
    tracker = create_tracker(interactive)
    with Database(config.db_path) as db, tracker:
        db.init_schema()

        papers = db.get_unenriched_papers()
        if not papers:
            logger.info("No papers to enrich")
            return

        from .enrichment.pwc import enrich_with_pwc
        from .enrichment.semantic_scholar import enrich_papers

        with tracker.stage("Semantic Scholar", total=len(papers)) as stage:
            papers = enrich_papers(papers, config, progress=stage)
            stage.set_detail(f"{len(papers)} enriched")

        with tracker.stage("Papers with Code") as stage:
            papers = enrich_with_pwc(papers, config)
            stage.set_detail(f"{len(papers)} checked")

        for paper in papers:
            db.update_enrichment(paper)

        logger.info("Enrichment complete")


def cmd_filter(args):
    """Filter papers using LLM relevance check."""
    config = get_config(args)
    if not config.llm.filter.enabled:
        logger.info("LLM filter is disabled in config")
        return

    interactive = _is_interactive()
    tracker = create_tracker(interactive)
    with Database(config.db_path) as db, tracker:
        db.init_schema()

        from .filter import PaperFilter

        papers = db.get_all_papers()
        if not papers:
            logger.info("No papers to filter")
            return

        with tracker.stage("Filter", total=len(papers)) as stage:
            filt = PaperFilter(config, db)
            relevant, rejected = filt.filter_papers(papers, progress=stage)
            stage.set_detail(f"{len(relevant)} relevant, {len(rejected)} rejected")
            stage.set_cost(filt.run_cost)

        logger.info(f"Filter: {len(relevant)} relevant, {len(rejected)} rejected")


def cmd_score(args):
    """Score all papers (quality signals only)."""
    config = get_config(args)
    interactive = _is_interactive()
    tracker = create_tracker(interactive)
    with Database(config.db_path) as db, tracker:
        db.init_schema()

        from .scoring import score_papers

        papers = db.get_all_papers()
        if not papers:
            logger.info("No papers to score")
            return

        with tracker.stage("Score", total=len(papers)) as stage:
            scored = score_papers(papers, config)
            for paper, scores in scored:
                db.upsert_scores(paper.db_id, scores)
            stage.set_detail(f"{len(scored)} papers scored")


def cmd_digest(args):
    """Generate and deliver digest."""
    config = get_config(args)
    interactive = _is_interactive()
    tracker = create_tracker(interactive)
    dry_run = getattr(args, "dry_run", False)
    with Database(config.db_path) as db:
        db.init_schema()
        with tracker:
            _cmd_digest_inner(config, db, tracker, dry_run=dry_run)


def cmd_run(args):
    """Full pipeline: fetch -> digest (includes filter, enrich, score, summarize, rank)."""
    config = get_config(args)
    interactive = _is_interactive()
    tracker = create_tracker(interactive)
    dry_run = getattr(args, "dry_run", False)
    with Database(config.db_path) as db:
        db.init_schema()
        with tracker:
            _cmd_fetch_inner(config, db, tracker)
            _cmd_digest_inner(config, db, tracker, dry_run=dry_run)
    logger.info("Pipeline run complete")


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _dir_size(path: Path) -> int:
    """Total size of all files in a directory."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def cmd_clean(args):
    """Remove generated data (database, digests, PWC cache)."""
    from rich.console import Console
    from rich.table import Table

    config = get_config(args)
    console = Console()

    clean_db = args.db
    clean_digests = args.digests
    clean_pwc = args.pwc
    # No flags = --all
    if not (clean_db or clean_digests or clean_pwc or args.all):
        clean_db = clean_digests = clean_pwc = True
    if args.all:
        clean_db = clean_digests = clean_pwc = True

    # Collect targets: (label, paths_to_delete, display_path, size)
    targets = []

    if clean_db:
        db_path = config.db_path
        db_files = [db_path, db_path.with_suffix(".db-wal"), db_path.with_suffix(".db-shm")]
        existing = [f for f in db_files if f.exists()]
        if existing:
            size = sum(f.stat().st_size for f in existing)
            targets.append(("Database", existing, str(db_path), size))

    if clean_digests:
        digest_dir = config.digest_dir
        if digest_dir.exists() and any(digest_dir.iterdir()):
            size = _dir_size(digest_dir)
            targets.append(("Digests", [digest_dir], str(digest_dir), size))

    if clean_pwc:
        pwc_path = config.pwc_path
        if pwc_path.exists():
            size = pwc_path.stat().st_size
            targets.append(("PWC cache", [pwc_path], str(pwc_path), size))

    if not targets:
        console.print("Nothing to clean.", style="dim")
        return

    # Show what will be deleted
    table = Table(title="Will be deleted", border_style="red")
    table.add_column("Target", style="bold")
    table.add_column("Path")
    table.add_column("Size", justify="right")
    for label, _, display_path, size in targets:
        table.add_row(label, display_path, _format_size(size))
    console.print(table)

    # Confirm
    if not args.yes:
        try:
            answer = input("Delete? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\nAborted.", style="dim")
            return
        if answer not in ("y", "yes"):
            console.print("Aborted.", style="dim")
            return

    # Delete
    for label, paths, _, _ in targets:
        for p in paths:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        console.print(f"  Removed {label}", style="green")

    console.print("Clean complete.", style="bold green")


def cmd_setup(args):
    """Interactive setup wizard."""
    from .setup import run_setup

    config_path = getattr(args, "config", "config.yaml")
    base_dir = Path(config_path).parent.resolve() if config_path != "config.yaml" else Path.cwd()
    run_setup(base_dir)


def cmd_serve(args):
    """Start the web dashboard."""
    config = get_config(args)

    from .web import run_server

    run_server(config)


def cmd_stats(args):
    """Show database statistics."""
    from rich.console import Console
    from rich.table import Table

    config = get_config(args)
    console = Console()

    with Database(config.db_path) as db:
        db.init_schema()

        paper_count = db.get_paper_count()
        scored_count = db.get_scored_count()
        digest_count = db.get_digest_count()

        db_table = Table(title="Database", show_header=False, border_style="dim")
        db_table.add_column(style="bold")
        db_table.add_column(justify="right")
        db_table.add_row("Path", str(config.db_path))
        db_table.add_row("Papers", str(paper_count))
        db_table.add_row("Scored", str(scored_count))
        db_table.add_row("Digests", str(digest_count))
        console.print(db_table)

        if config.llm.filter.enabled or config.llm.summarizer.enabled:
            stats = db.get_llm_stats()

            llm_table = Table(title=f"LLM Usage ({stats['month']})", show_header=False, border_style="dim")
            llm_table.add_column(style="bold")
            llm_table.add_column(justify="right")
            llm_table.add_row("Runs", str(stats["runs"]))
            llm_table.add_row("Papers summarized", str(stats["total_papers_summarized"]))
            llm_table.add_row("Total tokens", f"{stats['total_input_tokens']} in / {stats['total_output_tokens']} out")
            console.print(llm_table)

    if config.openai_admin_key:
        from .usage import fetch_openai_usage

        usage = fetch_openai_usage(config.openai_admin_key)
        if usage:
            oai_table = Table(
                title=f"OpenAI Account ({usage['month']})",
                show_header=False,
                border_style="dim",
            )
            oai_table.add_column(style="bold")
            oai_table.add_column(justify="right")
            oai_table.add_row("Cost", f"${usage['total_cost_usd']:.4f}")
            oai_table.add_row(
                "Tokens",
                f"{usage['total_input_tokens']:,} in / {usage['total_output_tokens']:,} out",
            )
            oai_table.add_row("Requests", f"{usage['total_requests']:,}")
            for model, m in sorted(usage["by_model"].items()):
                oai_table.add_row(
                    f"  {model}",
                    f"{m['input']:,} in / {m['output']:,} out / {m['requests']:,} req",
                )
            console.print(oai_table)


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

    # clean
    clean_parser = subparsers.add_parser("clean", parents=[common], help="Remove generated data")
    clean_parser.add_argument("--db", action="store_true", help="Remove database")
    clean_parser.add_argument("--digests", action="store_true", help="Remove generated digests")
    clean_parser.add_argument("--pwc", action="store_true", help="Remove PWC links cache")
    clean_parser.add_argument("--all", action="store_true", help="Remove everything (default if no flags)")
    clean_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")

    # setup
    subparsers.add_parser("setup", parents=[common], help="Interactive setup wizard")

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
        "clean": cmd_clean,
        "setup": cmd_setup,
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
