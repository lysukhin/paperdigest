"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TopicConfig:
    name: str
    primary_keywords: list[str]
    secondary_keywords: list[str] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=list)
    arxiv_categories: list[str] = field(default_factory=list)


@dataclass
class CollectionConfig:
    lookback_days: int = 7
    max_results: int = 200


@dataclass
class RelevanceWeights:
    primary_base: float = 0.5
    secondary_increment: float = 0.1
    secondary_cap: float = 0.3
    benchmark_increment: float = 0.1
    benchmark_cap: float = 0.2


@dataclass
class QualityWeights:
    w_venue: float = 0.25
    w_author: float = 0.20
    w_cite: float = 0.20
    w_code: float = 0.15
    w_fresh: float = 0.20


@dataclass
class ScoringConfig:
    alpha: float = 0.65
    relevance: RelevanceWeights = field(default_factory=RelevanceWeights)
    quality: QualityWeights = field(default_factory=QualityWeights)
    venue_tiers: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class CostControl:
    max_cost_per_run: float = 0.50
    max_cost_per_month: float = 10.00
    warn_at_percent: float = 80
    input_cost_per_1k: float = 0.00015
    output_cost_per_1k: float = 0.0006


@dataclass
class LLMConfig:
    enabled: bool = False
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    cost_control: CostControl = field(default_factory=CostControl)


@dataclass
class DigestConfig:
    top_n: int = 20
    summarize_top_n: int = 15
    output_dir: str = "data/digests"


@dataclass
class MarkdownDeliveryConfig:
    enabled: bool = True


@dataclass
class TelegramDeliveryConfig:
    enabled: bool = False


@dataclass
class DeliveryConfig:
    markdown: MarkdownDeliveryConfig = field(default_factory=MarkdownDeliveryConfig)
    telegram: TelegramDeliveryConfig = field(default_factory=TelegramDeliveryConfig)


@dataclass
class Config:
    topic: TopicConfig
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    digest: DigestConfig = field(default_factory=DigestConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    database: str = "data/papers.db"
    pwc_links_path: str = "data/pwc_links.json"

    # Resolved at load time
    base_dir: Path = field(default_factory=lambda: Path.cwd())

    @property
    def db_path(self) -> Path:
        return self.base_dir / self.database

    @property
    def pwc_path(self) -> Path:
        return self.base_dir / self.pwc_links_path

    @property
    def digest_dir(self) -> Path:
        return self.base_dir / self.digest.output_dir

    @property
    def llm_api_key(self) -> str | None:
        return os.environ.get("LLM_API_KEY")

    @property
    def semantic_scholar_api_key(self) -> str | None:
        return os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

    @property
    def telegram_bot_token(self) -> str | None:
        return os.environ.get("TELEGRAM_BOT_TOKEN")

    @property
    def telegram_chat_id(self) -> str | None:
        return os.environ.get("TELEGRAM_CHAT_ID")


def _build_topic(d: dict) -> TopicConfig:
    return TopicConfig(
        name=d["name"],
        primary_keywords=d["primary_keywords"],
        secondary_keywords=d.get("secondary_keywords", []),
        benchmarks=d.get("benchmarks", []),
        arxiv_categories=d.get("arxiv_categories", []),
    )


def _build_scoring(d: dict) -> ScoringConfig:
    rel = d.get("relevance", {})
    qual = d.get("quality", {})
    return ScoringConfig(
        alpha=d.get("alpha", 0.65),
        relevance=RelevanceWeights(**{k: v for k, v in rel.items()}),
        quality=QualityWeights(**{k: v for k, v in qual.items()}),
        venue_tiers=d.get("venue_tiers", {}),
    )


def _build_llm(d: dict) -> LLMConfig:
    cc = d.get("cost_control", {})
    return LLMConfig(
        enabled=d.get("enabled", False),
        model=d.get("model", "gpt-4o-mini"),
        base_url=d.get("base_url", "https://api.openai.com/v1"),
        cost_control=CostControl(**{k: v for k, v in cc.items()}),
    )


def _build_delivery(d: dict) -> DeliveryConfig:
    md = d.get("markdown", {})
    tg = d.get("telegram", {})
    return DeliveryConfig(
        markdown=MarkdownDeliveryConfig(enabled=md.get("enabled", True)),
        telegram=TelegramDeliveryConfig(enabled=tg.get("enabled", False)),
    )


def load_config(path: str | Path) -> Config:
    """Load and validate configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")

    if "topic" not in raw:
        raise ValueError("Config must include a 'topic' section")

    topic_raw = raw["topic"]
    if "name" not in topic_raw or "primary_keywords" not in topic_raw:
        raise ValueError("topic must have 'name' and 'primary_keywords'")

    if not topic_raw["primary_keywords"]:
        raise ValueError("primary_keywords must not be empty")

    base_dir = path.parent.resolve()
    coll = raw.get("collection", {})
    db_raw = raw.get("database", {})
    pwc_raw = raw.get("pwc", {})

    return Config(
        topic=_build_topic(raw["topic"]),
        collection=CollectionConfig(
            lookback_days=coll.get("lookback_days", 7),
            max_results=coll.get("max_results", 200),
        ),
        scoring=_build_scoring(raw.get("scoring", {})),
        llm=_build_llm(raw.get("llm", {})),
        digest=DigestConfig(
            top_n=raw.get("digest", {}).get("top_n", 20),
            summarize_top_n=raw.get("digest", {}).get("summarize_top_n", 15),
            output_dir=raw.get("digest", {}).get("output_dir", "data/digests"),
        ),
        delivery=_build_delivery(raw.get("delivery", {})),
        database=db_raw.get("path", "data/papers.db") if isinstance(db_raw, dict) else db_raw,
        pwc_links_path=pwc_raw.get("links_path", "data/pwc_links.json") if isinstance(pwc_raw, dict) else pwc_raw,
        base_dir=base_dir,
    )
