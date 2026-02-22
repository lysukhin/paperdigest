"""Papers with Code enrichment (via HuggingFace)."""

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

# PWC archive dataset hosted on HuggingFace (snapshot from July 2025)
HF_PARQUET_URL = (
    "https://huggingface.co/datasets/pwc-archive/links-between-paper-and-code"
    "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
# HuggingFace papers API for live lookups (papers newer than the snapshot)
HF_PAPERS_API = "https://huggingface.co/api/papers/{arxiv_id}"


def download_pwc_links(dest: Path):
    """Download paper-code links from HuggingFace PWC archive (parquet)."""
    logger.info(f"Downloading PWC links from HuggingFace to {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow is required for PWC download: pip install pyarrow")
        return

    try:
        resp = requests.get(HF_PARQUET_URL, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to download PWC links: {e}")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
        for chunk in resp.iter_content(chunk_size=65536):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        table = pq.read_table(
            tmp_path,
            columns=["paper_arxiv_id", "repo_url", "is_official"],
        )
    except Exception as e:
        logger.error(f"Failed to read parquet file: {e}")
        return
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Build a lookup keyed by arxiv ID
    lookup: dict[str, dict] = {}
    arxiv_ids = table.column("paper_arxiv_id").to_pylist()
    repo_urls = table.column("repo_url").to_pylist()
    officials = table.column("is_official").to_pylist()

    for aid, repo_url, is_official in zip(arxiv_ids, repo_urls, officials):
        if not aid or not repo_url:
            continue
        # Strip version suffix (e.g. "2401.12345v2" -> "2401.12345")
        aid = re.sub(r"v\d+$", "", aid)
        if aid not in lookup:
            lookup[aid] = {"code_url": repo_url, "is_official": bool(is_official)}
        elif is_official and not lookup[aid].get("is_official"):
            lookup[aid] = {"code_url": repo_url, "is_official": True}

    with open(dest, "w") as f:
        json.dump(lookup, f)

    logger.info(f"PWC lookup saved: {len(lookup)} entries")


def _hf_lookup(arxiv_id: str) -> dict | None:
    """Query HuggingFace papers API for a single paper's code link."""
    try:
        resp = requests.get(
            HF_PAPERS_API.format(arxiv_id=arxiv_id), timeout=5
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        repo = data.get("githubRepo")
        if repo:
            return {"code_url": repo, "is_official": True}
    except (requests.RequestException, ValueError):
        pass
    return None


def load_pwc_lookup(path: Path) -> dict[str, dict]:
    """Load the pre-processed PWC lookup."""
    if not path.exists():
        logger.warning(f"PWC links not found at {path}. Run 'init' to download.")
        return {}
    with open(path) as f:
        return json.load(f)


def enrich_with_pwc(papers: list[Paper], config: Config) -> list[Paper]:
    """Enrich papers with code links from PWC dump + HuggingFace API fallback."""
    lookup = load_pwc_lookup(config.pwc_path)

    matched = 0
    hf_hits = 0
    for paper in papers:
        if paper.code_url:
            matched += 1
            continue

        entry = lookup.get(paper.arxiv_id)
        if not entry and not paper.arxiv_id.startswith(("dblp:", "nvidia:", "waymo:")):
            # Try live HF API for arXiv papers not in the static dump
            entry = _hf_lookup(paper.arxiv_id)
            if entry:
                hf_hits += 1

        if entry:
            paper.code_url = entry.get("code_url")
            paper.code_official = entry.get("is_official", False)
            matched += 1

    hf_msg = f" ({hf_hits} from HuggingFace API)" if hf_hits else ""
    logger.info(f"PWC enrichment: {matched}/{len(papers)} papers have code{hf_msg}")
    return papers
