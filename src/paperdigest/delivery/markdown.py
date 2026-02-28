"""Markdown file delivery."""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import FileSystemLoader
from jinja2.sandbox import SandboxedEnvironment

from ..config import Config
from ..models import Digest

logger = logging.getLogger(__name__)


def render_digest(digest: Digest, config: Config) -> str:
    """Render a digest to markdown using the Jinja2 template."""
    template_dir = Path(__file__).parent.parent / "templates"
    env = SandboxedEnvironment(loader=FileSystemLoader(str(template_dir)), trim_blocks=True, lstrip_blocks=True)
    template = env.get_template("digest.md.j2")
    return template.render(digest=digest)


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
