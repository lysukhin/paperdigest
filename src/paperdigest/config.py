"""Configuration loading and validation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TopicConfig:
    name: str
    primary_keywords: list[str]
    secondary_keywords: list[str] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=list)
    arxiv_categories: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class BlogsConfig:
    enabled: bool = False
    sources: list[str] = field(default_factory=lambda: ["nvidia", "waymo", "wayve"])


@dataclass
class ConferencesConfig:
    enabled: bool = False
    venues: list[str] = field(default_factory=list)
    years_back: int = 1


@dataclass
class CollectionConfig:
    lookback_days: int = 7
    max_results: int = 200
    blogs: BlogsConfig = field(default_factory=BlogsConfig)
    conferences: ConferencesConfig = field(default_factory=ConferencesConfig)


@dataclass
class EnrichmentConfig:
    semantic_scholar_enabled: bool = True


@dataclass
class QualityWeights:
    w_venue: float = 0.25
    w_author: float = 0.20
    w_cite: float = 0.20
    w_code: float = 0.15
    w_fresh: float = 0.20


@dataclass
class ScoringConfig:
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
class FilterLLMConfig:
    enabled: bool = False
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float | None = None
    max_completion_tokens: int | None = 256
    extra_instructions: str | None = None
    cost_control: CostControl = field(default_factory=CostControl)


@dataclass
class SummarizerLLMConfig:
    enabled: bool = False
    model: str = "gpt-5-nano-2025-08-07"
    base_url: str = "https://api.openai.com/v1"
    temperature: float | None = None
    max_completion_tokens: int | None = 16384
    max_text_chars: int = 50000
    language: str = "Russian"
    extra_instructions: str | None = None
    cost_control: CostControl = field(default_factory=CostControl)


@dataclass
class LLMConfig:
    filter: FilterLLMConfig = field(default_factory=FilterLLMConfig)
    summarizer: SummarizerLLMConfig = field(default_factory=SummarizerLLMConfig)


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
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    public_url: str | None = None


@dataclass
class DeliveryConfig:
    markdown: MarkdownDeliveryConfig = field(default_factory=MarkdownDeliveryConfig)
    telegram: TelegramDeliveryConfig = field(default_factory=TelegramDeliveryConfig)


@dataclass
class Config:
    topic: TopicConfig
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    digest: DigestConfig = field(default_factory=DigestConfig)
    delivery: DeliveryConfig = field(default_factory=DeliveryConfig)
    web: WebConfig = field(default_factory=WebConfig)
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
    def openai_admin_key(self) -> str | None:
        return os.environ.get("OPENAI_ADMIN_KEY")

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
        description=d.get("description", ""),
    )


def _build_scoring(d: dict) -> ScoringConfig:
    qual = d.get("quality", {})
    return ScoringConfig(
        quality=QualityWeights(**qual),
        venue_tiers=d.get("venue_tiers", {}),
    )


def _build_llm(d: dict) -> LLMConfig:
    filter_raw = d.get("filter", {})
    summ_raw = d.get("summarizer", {})
    filter_cc = filter_raw.get("cost_control", {})
    summ_cc = summ_raw.get("cost_control", {})
    return LLMConfig(
        filter=FilterLLMConfig(
            enabled=filter_raw.get("enabled", False),
            model=filter_raw.get("model", "gpt-4o-mini"),
            base_url=filter_raw.get("base_url", "https://api.openai.com/v1"),
            temperature=filter_raw.get("temperature"),
            max_completion_tokens=filter_raw.get("max_completion_tokens", 256),
            extra_instructions=filter_raw.get("extra_instructions"),
            cost_control=CostControl(**filter_cc),
        ),
        summarizer=SummarizerLLMConfig(
            enabled=summ_raw.get("enabled", False),
            model=summ_raw.get("model", "gpt-5-nano-2025-08-07"),
            base_url=summ_raw.get("base_url", "https://api.openai.com/v1"),
            temperature=summ_raw.get("temperature"),
            max_completion_tokens=summ_raw.get("max_completion_tokens", 16384),
            max_text_chars=summ_raw.get("max_text_chars", 50000),
            language=summ_raw.get("language", "Russian"),
            extra_instructions=summ_raw.get("extra_instructions"),
            cost_control=CostControl(**summ_cc),
        ),
    )


def _build_delivery(d: dict) -> DeliveryConfig:
    md = d.get("markdown", {})
    tg = d.get("telegram", {})
    return DeliveryConfig(
        markdown=MarkdownDeliveryConfig(enabled=md.get("enabled", True)),
        telegram=TelegramDeliveryConfig(enabled=tg.get("enabled", False)),
    )


def _normalize_url(raw: str) -> str | None:
    url = raw.strip().rstrip("/")
    if not url:
        return None
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url


def _load_env_file(env_path: Path) -> None:
    """Load variables from a .env file into os.environ.

    Existing environment variables are NOT overwritten.
    Supports KEY=VALUE, quoted values, comments, and blank lines.
    """
    if not env_path.is_file():
        return

    loaded = 0
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        # Strip matching quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]

        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1

    if loaded:
        logger.debug(f"Loaded {loaded} variable(s) from {env_path}")


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

    # Load .env from the same directory as config.yaml
    _load_env_file(base_dir / ".env")

    coll = raw.get("collection", {})
    enrich_raw = raw.get("enrichment", {}) or {}
    ss_raw = enrich_raw.get("semantic_scholar", {}) or {}
    db_raw = raw.get("database", {})
    pwc_raw = raw.get("pwc", {})

    scoring = _build_scoring(raw.get("scoring", {}))

    # Validate scoring config
    qw = scoring.quality
    weight_sum = qw.w_venue + qw.w_author + qw.w_cite + qw.w_code + qw.w_fresh
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(f"Quality weights must sum to 1.0, got {weight_sum:.4f}")

    web_raw = raw.get("web") or {}

    return Config(
        topic=_build_topic(raw["topic"]),
        collection=CollectionConfig(
            lookback_days=coll.get("lookback_days", 7),
            max_results=coll.get("max_results", 200),
            blogs=BlogsConfig(
                enabled=coll.get("blogs", {}).get("enabled", False),
                sources=coll.get("blogs", {}).get("sources", ["nvidia", "waymo", "wayve"]),
            ),
            conferences=ConferencesConfig(
                enabled=coll.get("conferences", {}).get("enabled", False),
                venues=coll.get("conferences", {}).get("venues", []),
                years_back=coll.get("conferences", {}).get("years_back", 1),
            ),
        ),
        enrichment=EnrichmentConfig(
            semantic_scholar_enabled=ss_raw.get("enabled", True),
        ),
        scoring=scoring,
        llm=_build_llm(raw.get("llm", {})),
        digest=DigestConfig(
            top_n=raw.get("digest", {}).get("top_n", 20),
            summarize_top_n=raw.get("digest", {}).get("summarize_top_n", 15),
            output_dir=raw.get("digest", {}).get("output_dir", "data/digests"),
        ),
        delivery=_build_delivery(raw.get("delivery", {})),
        web=WebConfig(
            host=web_raw.get("host", "127.0.0.1"),
            port=web_raw.get("port", 8000),
            public_url=_normalize_url(web_raw.get("public_url", "")),
        ),
        database=db_raw.get("path", "data/papers.db") if isinstance(db_raw, dict) else db_raw,
        pwc_links_path=pwc_raw.get("links_path", "data/pwc_links.json") if isinstance(pwc_raw, dict) else pwc_raw,
        base_dir=base_dir,
    )
