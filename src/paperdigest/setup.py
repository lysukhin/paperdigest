"""Setup wizard for paperdigest — generates config files and runs interactive onboarding."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation functions (pure, testable — no I/O prompts)
# ---------------------------------------------------------------------------


def generate_config(
    path: Path,
    *,
    topic_name: str,
    topic_description: str,
    primary_keywords: list[str],
    arxiv_categories: list[str],
    language: str,
    domain: str | None,
) -> None:
    """Write a config.yaml to *path* from the given parameters.

    Uses gpt-4o-mini as the default model for both filter and summarizer.
    Sets ``web.host`` to ``"0.0.0.0"`` (required for Docker).
    If *domain* is provided, sets ``web.public_url`` to ``https://{domain}``.
    """
    web_section: dict = {"host": "0.0.0.0", "port": 8000}
    if domain:
        web_section["public_url"] = f"https://{domain}"

    data: dict = {
        "topic": {
            "name": topic_name,
            "description": topic_description,
            "primary_keywords": primary_keywords,
            "arxiv_categories": arxiv_categories,
        },
        "collection": {
            "lookback_days": 7,
            "max_results": 200,
        },
        "llm": {
            "filter": {
                "enabled": True,
                "model": "gpt-4o-mini",
            },
            "summarizer": {
                "enabled": True,
                "model": "gpt-4o-mini",
                "language": language,
            },
        },
        "digest": {
            "top_n": 20,
            "summarize_top_n": 10,
            "output_dir": "data/digests",
        },
        "delivery": {
            "markdown": {"enabled": True},
            "telegram": {"enabled": False},
        },
        "database": {"path": "data/papers.db"},
        "web": web_section,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)


def generate_env(
    path: Path,
    *,
    llm_api_key: str,
    semantic_scholar_key: str,
    openai_admin_key: str,
    telegram_bot_token: str,
    telegram_chat_id: str,
) -> None:
    """Write a ``.env`` file to *path*.

    Only non-empty values are written.  Format: ``KEY=value`` one per line.
    """
    entries = [
        ("LLM_API_KEY", llm_api_key),
        ("SEMANTIC_SCHOLAR_API_KEY", semantic_scholar_key),
        ("OPENAI_ADMIN_KEY", openai_admin_key),
        ("TELEGRAM_BOT_TOKEN", telegram_bot_token),
        ("TELEGRAM_CHAT_ID", telegram_chat_id),
    ]

    lines = [f"{key}={value}" for key, value in entries if value]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def generate_caddyfile(path: Path, *, domain: str | None) -> None:
    """Write a Caddyfile to *path*.

    If *domain* is given it is used as the host block; otherwise ``:80`` is used.
    Always proxies to ``web:8000``.
    """
    host = domain if domain else ":80"
    content = f"{host} {{\n    reverse_proxy web:8000\n}}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def generate_crontab(path: Path, *, schedule: str) -> None:
    """Write a crontab file to *path*.

    Format: ``{schedule} paperdigest run --config /app/config.yaml >> /app/data/cron.log 2>&1``
    """
    line = f"{schedule} paperdigest run --config /app/config.yaml >> /app/data/cron.log 2>&1\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line)


# ---------------------------------------------------------------------------
# Interactive wizard helpers
# ---------------------------------------------------------------------------


def _prompt(label: str, default: str = "") -> str:
    """Prompt the user for a value, showing a default in brackets."""
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(130)
    return value or default


def _prompt_yn(label: str, default: bool = True) -> bool:
    """Prompt the user for a yes/no answer."""
    hint = "Y/n" if default else "y/N"
    try:
        value = input(f"{label} [{hint}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(130)
    if not value:
        return default
    return value in ("y", "yes")


def _prompt_list(label: str, default: str = "") -> list[str]:
    """Prompt for a comma-separated list and return split values."""
    raw = _prompt(label, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _test_llm_connection(api_key: str, base_url: str = "https://api.openai.com/v1") -> bool:
    """Test the LLM connection by listing available models."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        client.models.list()
        return True
    except Exception as exc:
        logger.warning(f"LLM connection test failed: {exc}")
        return False


def _test_telegram(bot_token: str, chat_id: str) -> bool:
    """Send a test message via Telegram."""
    try:
        import urllib.request
        import json

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": "paperdigest setup: test message"}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning(f"Telegram test failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main interactive wizard
# ---------------------------------------------------------------------------


def run_setup(base_dir: Path) -> None:
    """Run the interactive setup wizard.

    Prompts the user through topic configuration, API keys, Telegram,
    schedule, and domain.  Writes config files, initialises the database,
    and optionally downloads PWC links.
    """
    print("=== paperdigest setup wizard ===\n")

    # --- Topic ---
    topic_name = _prompt("Research topic name", "Autonomous Driving")
    topic_description = _prompt(
        "Topic description (for LLM filter)",
        "Papers about autonomous driving and related topics",
    )
    primary_keywords = _prompt_list("Primary keywords (comma-separated)", "autonomous driving, self-driving")
    arxiv_categories = _prompt_list("arXiv categories (comma-separated)", "cs.CV, cs.AI, cs.RO")
    language = _prompt("Summary language", "English")

    # --- API keys ---
    print("\n--- API Keys ---")
    llm_api_key = _prompt("OpenAI API key (LLM_API_KEY)")
    semantic_scholar_key = _prompt("Semantic Scholar API key (optional)")
    openai_admin_key = _prompt("OpenAI Admin key (optional, for usage stats)")

    # Test LLM connection
    if llm_api_key:
        print("Testing LLM connection...", end=" ", flush=True)
        if _test_llm_connection(llm_api_key):
            print("OK")
        else:
            print("FAILED (continuing anyway)")

    # --- Telegram ---
    print("\n--- Telegram (optional) ---")
    enable_telegram = _prompt_yn("Enable Telegram delivery?", default=False)
    telegram_bot_token = ""
    telegram_chat_id = ""
    if enable_telegram:
        telegram_bot_token = _prompt("Telegram bot token")
        telegram_chat_id = _prompt("Telegram chat ID")
        if telegram_bot_token and telegram_chat_id:
            print("Sending test message...", end=" ", flush=True)
            if _test_telegram(telegram_bot_token, telegram_chat_id):
                print("OK")
            else:
                print("FAILED (continuing anyway)")

    # --- Schedule ---
    print("\n--- Cron Schedule ---")
    schedule = _prompt("Cron schedule (digest generation, UTC)", "0 9 * * *")

    # --- Domain ---
    print("\n--- Domain ---")
    domain = _prompt("Domain for web dashboard (optional, e.g. digest.example.com)")
    domain = domain if domain else None

    # --- Generate files ---
    print("\n--- Generating files ---")

    config_path = base_dir / "config.yaml"
    generate_config(
        config_path,
        topic_name=topic_name,
        topic_description=topic_description,
        primary_keywords=primary_keywords,
        arxiv_categories=arxiv_categories,
        language=language,
        domain=domain,
    )
    print("  config.yaml")

    # Enable Telegram in config if user opted in
    if enable_telegram and telegram_bot_token:
        with open(config_path) as fh:
            data = yaml.safe_load(fh)
        data["delivery"]["telegram"]["enabled"] = True
        with open(config_path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    env_path = base_dir / ".env"
    generate_env(
        env_path,
        llm_api_key=llm_api_key,
        semantic_scholar_key=semantic_scholar_key,
        openai_admin_key=openai_admin_key,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
    )
    print("  .env")

    caddyfile_path = base_dir / "Caddyfile"
    generate_caddyfile(caddyfile_path, domain=domain)
    print("  Caddyfile")

    crontab_path = base_dir / "crontab"
    generate_crontab(crontab_path, schedule=schedule)
    print("  crontab")

    # --- Create data directories ---
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "digests").mkdir(parents=True, exist_ok=True)
    print("  data/ directories")

    # --- Init DB ---
    from .db import Database

    db_path = data_dir / "papers.db"
    with Database(db_path) as db:
        db.init_schema()
    print("  database initialized")

    # --- PWC links (optional) ---
    if _prompt_yn("Download Papers with Code links? (~400 MB)", default=False):
        try:
            from .enrichment.pwc import download_pwc_links

            pwc_path = data_dir / "pwc_links.json"
            print("Downloading PWC links...", flush=True)
            download_pwc_links(pwc_path)
            print("  PWC links downloaded")
        except Exception as exc:
            print(f"  PWC download failed: {exc} (skipping)")

    # --- Summary ---
    print("\n=== Setup complete! ===")
    print(f"\nGenerated files in {base_dir}:")
    print("  config.yaml   — pipeline configuration")
    print("  .env          — API keys and secrets")
    print("  Caddyfile     — reverse proxy config")
    print("  crontab       — digest schedule")
    print("\nNext steps:")
    print("  docker compose up -d        # start all services")
    print("  docker compose logs -f       # watch logs")
    if domain:
        print(f"  Point DNS for {domain} -> your server IP")
    print()
