"""Telegram Bot API delivery (raw HTTP, no library)."""

from __future__ import annotations

import logging

import requests

from ..config import Config
from ..models import Digest

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LENGTH = 4096


def _format_telegram_message(digest: Digest, config: Config) -> str:
    """Format digest as a Telegram-friendly message (Markdown)."""
    lines = [
        f"*📚 Paper Digest: {digest.topic_name}*",
        f"_{digest.date.strftime('%Y-%m-%d')}_ | {digest.total_collected} collected, {len(digest.entries)} ranked\n",
    ]

    for entry in digest.entries[:10]:  # Telegram messages have length limits
        p = entry.paper
        s = entry.scores
        line = f"*{entry.rank}.* [{p.title[:80]}](https://arxiv.org/abs/{p.arxiv_id})"
        line += f"\n   Score: {s.final:.2f} | Cites: {p.citations or 0}"
        if p.code_url:
            line += f" | [Code]({p.code_url})"
        if entry.summary and entry.summary.one_liner:
            line += f"\n   _{entry.summary.one_liner}_"
        lines.append(line)

    return "\n".join(lines)


def deliver_telegram(digest: Digest, config: Config) -> bool:
    """Send digest via Telegram Bot API."""
    token = config.telegram_bot_token
    chat_id = config.telegram_chat_id

    if not token or not chat_id:
        logger.warning("Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return False

    message = _format_telegram_message(digest, config)

    # Truncate if too long
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[: MAX_MESSAGE_LENGTH - 20] + "\n\n_...truncated_"

    url = TELEGRAM_API.format(token=token)
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Telegram message sent successfully")
        return True
    except requests.RequestException:
        logger.exception("Failed to send Telegram message")
        return False
