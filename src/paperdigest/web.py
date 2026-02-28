"""Web dashboard for paperdigest."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from .config import Config

logger = logging.getLogger("paperdigest.web")


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="paperdigest", docs_url=None, redoc_url=None)
    template_dir = Path(__file__).parent / "templates" / "web"
    templates = Jinja2Templates(directory=str(template_dir))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        digests = _list_digests(config)
        stats = _get_stats(config)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "digests": digests,
                "stats": stats,
                "topic": config.topic.name,
            },
        )

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

    return app


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


def _parse_digest_meta(path: Path) -> dict | None:
    match = re.search(r"digest_(\d+)", path.name)
    if not match:
        return None

    number = int(match.group(1))
    content = path.read_text()

    topic = ""
    papers_count = 0
    first_papers: list[str] = []

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


def _render_digest(config: Config, number: int) -> str | None:
    import markdown

    path = config.digest_dir / f"digest_{number:03d}.md"
    if not path.exists():
        return None

    text = path.read_text()
    # The markdown lib skips markdown inside HTML blocks like <details>.
    # Inject markdown="1" so md_in_html can process the inner content.
    text = text.replace("<details>", '<details markdown="1">')
    md = markdown.Markdown(extensions=["tables", "fenced_code", "md_in_html"])
    return md.convert(text)


def _get_stats(config: Config) -> dict:
    from .db import Database

    stats = {"papers": 0, "scored": 0, "digests": 0}
    if not config.db_path.exists():
        return stats

    try:
        with Database(config.db_path) as db:
            db.init_schema()
            stats["papers"] = db.get_paper_count()
            stats["scored"] = db.get_scored_count()
            stats["digests"] = db.get_digest_count()
    except Exception:
        logger.debug("Could not read DB stats", exc_info=True)

    from .usage import read_usage_cache

    usage = read_usage_cache(config.db_path.parent)
    if usage:
        stats["openai_usage"] = usage

    return stats


def run_server(config: Config):
    import os

    import uvicorn

    app = create_app(config)
    host = os.environ.get("WEB_HOST", config.web.host)
    port = int(os.environ.get("WEB_PORT", config.web.port))

    print(f"  paperdigest dashboard")
    print(f"  http://{host}:{port}")
    print()
    uvicorn.run(app, host=host, port=port, log_level="info")
