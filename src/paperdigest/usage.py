"""Fetch real usage data from the OpenAI Usage & Costs APIs."""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger("paperdigest.usage")


def fetch_openai_usage(admin_key: str) -> dict | None:
    """Fetch this month's token usage and costs from the OpenAI API.

    Requires an OpenAI Admin Key (OPENAI_ADMIN_KEY env var).
    Returns None on any error.
    """
    try:
        now = datetime.now(timezone.utc)
        start_of_month = int(
            datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp()
        )

        tokens = _fetch_completions_usage(admin_key, start_of_month)
        costs = _fetch_costs(admin_key, start_of_month)

        return {
            "month": f"{now.year:04d}-{now.month:02d}",
            "total_cost_usd": costs,
            "total_input_tokens": tokens["input"],
            "total_output_tokens": tokens["output"],
            "total_requests": tokens["requests"],
            "by_model": tokens["by_model"],
        }
    except Exception:
        logger.debug("Failed to fetch OpenAI usage", exc_info=True)
        return None


def _fetch_completions_usage(admin_key: str, start_time: int) -> dict:
    """Fetch token usage from GET /v1/organization/usage/completions."""
    total_input = 0
    total_output = 0
    total_requests = 0
    by_model: dict[str, dict] = {}

    page = None
    while True:
        params = (
            f"start_time={start_time}"
            f"&bucket_width=1d"
            f"&group_by[]=model"
        )
        if page:
            params += f"&page={page}"

        url = f"https://api.openai.com/v1/organization/usage/completions?{params}"
        data = _api_get(url, admin_key)

        for bucket in data.get("data", []):
            for result in bucket.get("results", []):
                inp = result.get("input_tokens", 0)
                out = result.get("output_tokens", 0)
                reqs = result.get("num_model_requests", 0)
                model = result.get("model") or "unknown"

                total_input += inp
                total_output += out
                total_requests += reqs

                if model not in by_model:
                    by_model[model] = {"input": 0, "output": 0, "requests": 0}
                by_model[model]["input"] += inp
                by_model[model]["output"] += out
                by_model[model]["requests"] += reqs

        if not data.get("has_more"):
            break
        page = data.get("next_page")

    return {
        "input": total_input,
        "output": total_output,
        "requests": total_requests,
        "by_model": by_model,
    }


def _fetch_costs(admin_key: str, start_time: int) -> float:
    """Fetch actual USD costs from GET /v1/organization/costs."""
    total = 0.0

    page = None
    while True:
        params = f"start_time={start_time}&bucket_width=1d"
        if page:
            params += f"&page={page}"

        url = f"https://api.openai.com/v1/organization/costs?{params}"
        data = _api_get(url, admin_key)

        for bucket in data.get("data", []):
            for result in bucket.get("results", []):
                amount = result.get("amount", {})
                total += float(amount.get("value", 0))

        if not data.get("has_more"):
            break
        page = data.get("next_page")

    return total


def _api_get(url: str, admin_key: str) -> dict:
    """Make an authenticated GET request to the OpenAI API."""
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {admin_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())
