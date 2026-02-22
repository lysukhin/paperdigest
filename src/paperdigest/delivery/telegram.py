"""Telegram Bot API delivery (raw HTTP, no library)."""

from __future__ import annotations

import json
import logging

import requests

from ..config import Config
from ..models import Digest

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LENGTH = 4096
TOP_N_TELEGRAM = 5


def _escape_markdown(text: str) -> str:
    """Escape Telegram Markdown special characters."""
    for ch in ('_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'):
        text = text.replace(ch, f'\\{ch}')
    return text


def _format_telegram_message(digest: Digest, config: Config) -> str:
    """Format digest as a compact Telegram message (MarkdownV2)."""
    topic = _escape_markdown(digest.topic_name)
    date_str = _escape_markdown(digest.date.strftime("%Y-%m-%d"))
    n_ranked = len(digest.entries)

    lines = [
        f"*Paper Digest: {topic}*",
        f"_{date_str}_ \\| {digest.total_collected} collected, {n_ranked} ranked",
        "",
    ]

    for entry in digest.entries[:TOP_N_TELEGRAM]:
        title = _escape_markdown(entry.paper.title[:80])
        lines.append(f"*{entry.rank}\\.*  {title}")
        if entry.summary and entry.summary.one_liner:
            one_liner = _escape_markdown(entry.summary.one_liner)
            lines.append(f"  _\\> {one_liner}_")

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
        message = message[: MAX_MESSAGE_LENGTH - 20] + "\n\n_\\.\\.\\.truncated_"

    url = TELEGRAM_API.format(token=token)
    payload: dict = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True,
    }

    # Add inline button linking to web digest
    public_url = config.web.public_url
    if public_url:
        date_str = digest.date.strftime("%Y-%m-%d")
        digest_url = f"{public_url}/digest/{date_str}"
        payload["reply_markup"] = json.dumps({
            "inline_keyboard": [[
                {"text": "\U0001f4d6 View Full Digest", "url": digest_url}
            ]]
        })

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram message sent successfully")
        return True
    except requests.RequestException as e:
        sanitized = str(e).replace(token, "<REDACTED>")
        logger.error(f"Failed to send Telegram message: {sanitized}")
        return False
