"""Papers with Code enrichment."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path

import requests

from ..config import Config
from ..models import Paper

logger = logging.getLogger(__name__)

PWC_LINKS_URL = "https://paperswithcode.com/media/about/links-between-papers-and-code.json.gz"


def download_pwc_links(dest: Path):
    """Download the PWC links dump (gzipped JSON)."""
    import gzip
    import shutil

    logger.info(f"Downloading PWC links to {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(PWC_LINKS_URL, stream=True, timeout=120)
    resp.raise_for_status()

    # Stream to temp file to avoid loading entire response into memory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as tmp_gz:
        for chunk in resp.iter_content(chunk_size=8192):
            tmp_gz.write(chunk)
        tmp_gz_path = tmp_gz.name

    try:
        with gzip.open(tmp_gz_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    finally:
        Path(tmp_gz_path).unlink(missing_ok=True)

    # Build a lookup keyed by arxiv paper URL
    lookup: dict[str, dict] = {}
    for entry in data:
        paper_url = entry.get("paper_url", "")
        # Extract arxiv ID from URL like https://arxiv.org/abs/2401.12345
        if "arxiv.org" in paper_url:
            aid = paper_url.rstrip("/").split("/")[-1]
            aid = re.sub(r'v\d+$', '', aid)
            if aid not in lookup:
                lookup[aid] = {
                    "code_url": entry.get("repo_url"),
                    "is_official": entry.get("is_official", False),
                }
            elif entry.get("is_official", False) and not lookup[aid].get("is_official"):
                # Prefer official repos
                lookup[aid] = {
                    "code_url": entry.get("repo_url"),
                    "is_official": True,
                }

    with open(dest, "w") as f:
        json.dump(lookup, f)

    logger.info(f"PWC lookup saved: {len(lookup)} entries")


def load_pwc_lookup(path: Path) -> dict[str, dict]:
    """Load the pre-processed PWC lookup."""
    if not path.exists():
        logger.warning(f"PWC links not found at {path}. Run 'init' to download.")
        return {}
    with open(path) as f:
        return json.load(f)


def enrich_with_pwc(papers: list[Paper], config: Config) -> list[Paper]:
    """Enrich papers with code availability from PWC dump."""
    lookup = load_pwc_lookup(config.pwc_path)
    if not lookup:
        return papers

    matched = 0
    for paper in papers:
        entry = lookup.get(paper.arxiv_id)
        if entry:
            paper.code_url = paper.code_url or entry.get("code_url")
            paper.code_official = paper.code_official or entry.get("is_official", False)
            matched += 1

    logger.info(f"PWC enrichment: {matched}/{len(papers)} papers have code")
    return papers
