"""PDF download and text extraction."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def fetch_paper_text(pdf_url: str, max_chars: int = 50000) -> str | None:
    """Download a PDF and extract its text content.

    Returns the extracted text truncated to max_chars, or None on failure.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.warning("pymupdf not installed. Install with: pip install pymupdf")
        return None

    try:
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
    except Exception:
        logger.warning(f"Failed to download PDF: {pdf_url}")
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(resp.content)
            tmp.flush()

            doc = fitz.open(tmp.name)
            pages = []
            char_count = 0
            for page in doc:
                text = page.get_text()
                pages.append(text)
                char_count += len(text)
                if char_count >= max_chars:
                    break
            doc.close()

        full_text = "\n".join(pages)
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]

        if not full_text.strip():
            logger.warning(f"No text extracted from PDF: {pdf_url}")
            return None

        return full_text
    except Exception:
        logger.warning(f"Failed to extract text from PDF: {pdf_url}", exc_info=True)
        return None
