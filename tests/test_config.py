"""Tests for config loading."""

from pathlib import Path

import pytest
import yaml

from paperdigest.config import load_config

MINIMAL_CONFIG = """\
topic:
  name: Test Topic
  primary_keywords:
    - keyword one
"""


def _write_config(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p


class TestConfigLoading:
    def test_valid_config(self, tmp_path):
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test Topic",
                "primary_keywords": ["keyword one"],
            }
        })
        config = load_config(path)
        assert config.topic.name == "Test Topic"
        assert config.topic.primary_keywords == ["keyword one"]
        assert config.topic.description == ""  # default

    def test_missing_topic_raises(self, tmp_path):
        path = _write_config(tmp_path, {"scoring": {}})
        with pytest.raises(ValueError, match="topic"):
            load_config(path)

    def test_missing_name_raises(self, tmp_path):
        path = _write_config(tmp_path, {"topic": {"primary_keywords": ["a"]}})
        with pytest.raises(ValueError, match="name"):
            load_config(path)

    def test_empty_keywords_raises(self, tmp_path):
        path = _write_config(tmp_path, {
            "topic": {"name": "Test", "primary_keywords": []}
        })
        with pytest.raises(ValueError, match="primary_keywords"):
            load_config(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_custom_scoring(self, tmp_path):
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "scoring": {
                "quality": {
                    "w_venue": 0.40,
                    "w_code": 0.25,
                    "w_fresh": 0.35,
                },
            },
        })
        config = load_config(path)
        assert config.scoring.quality.w_venue == 0.40
        assert not hasattr(config.scoring, "alpha")
        assert not hasattr(config.scoring, "relevance")

    def test_llm_defaults_disabled(self, tmp_path):
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
        })
        config = load_config(path)
        assert config.llm.filter.enabled is False
        assert config.llm.summarizer.enabled is False

    def test_llm_cost_control(self, tmp_path):
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "llm": {
                "summarizer": {
                    "enabled": True,
                    "model": "gpt-5-nano-2025-08-07",
                    "cost_control": {
                        "max_cost_per_run": 1.0,
                        "max_cost_per_month": 20.0,
                    },
                },
            },
        })
        config = load_config(path)
        assert config.llm.summarizer.enabled is True
        assert config.llm.summarizer.cost_control.max_cost_per_run == 1.0
        assert config.llm.summarizer.cost_control.max_cost_per_month == 20.0

    def test_invalid_quality_weights_raises(self, tmp_path):
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "scoring": {
                "quality": {"w_venue": 0.9},
            },
        })
        with pytest.raises(ValueError, match="Quality weights"):
            load_config(path)


class TestTopicDescription:
    def test_topic_description_default(self, tmp_path):
        """topic.description defaults to empty string."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            }
        })
        config = load_config(path)
        assert config.topic.description == ""

    def test_topic_description_loaded(self, tmp_path):
        """topic.description is loaded from config."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Autonomous Driving",
                "primary_keywords": ["self-driving"],
                "description": "Vision-language models for autonomous driving",
            }
        })
        config = load_config(path)
        assert config.topic.description == "Vision-language models for autonomous driving"


class TestSplitLLMConfig:
    def test_filter_defaults(self, tmp_path):
        """Filter LLM config has correct defaults."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            }
        })
        config = load_config(path)
        assert config.llm.filter.enabled is False
        assert config.llm.filter.model == "gpt-4o-mini"
        assert config.llm.filter.base_url == "https://api.openai.com/v1"
        assert config.llm.filter.temperature is None
        assert config.llm.filter.max_completion_tokens == 256

    def test_summarizer_defaults(self, tmp_path):
        """Summarizer LLM config has correct defaults."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            }
        })
        config = load_config(path)
        assert config.llm.summarizer.enabled is False
        assert config.llm.summarizer.model == "gpt-5-nano-2025-08-07"
        assert config.llm.summarizer.base_url == "https://api.openai.com/v1"
        assert config.llm.summarizer.temperature is None
        assert config.llm.summarizer.max_completion_tokens == 16384
        assert config.llm.summarizer.max_text_chars == 50000

    def test_filter_custom_config(self, tmp_path):
        """Filter LLM config can be customized."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "llm": {
                "filter": {
                    "enabled": True,
                    "model": "gpt-4o",
                    "temperature": 0.2,
                    "max_completion_tokens": 512,
                    "cost_control": {
                        "max_cost_per_run": 0.25,
                        "max_cost_per_month": 5.0,
                    },
                },
            },
        })
        config = load_config(path)
        assert config.llm.filter.enabled is True
        assert config.llm.filter.model == "gpt-4o"
        assert config.llm.filter.temperature == 0.2
        assert config.llm.filter.max_completion_tokens == 512
        assert config.llm.filter.cost_control.max_cost_per_run == 0.25
        assert config.llm.filter.cost_control.max_cost_per_month == 5.0

    def test_summarizer_custom_config(self, tmp_path):
        """Summarizer LLM config can be customized."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "llm": {
                "summarizer": {
                    "enabled": True,
                    "model": "gpt-4o",
                    "max_text_chars": 30000,
                    "cost_control": {
                        "max_cost_per_run": 2.0,
                    },
                },
            },
        })
        config = load_config(path)
        assert config.llm.summarizer.enabled is True
        assert config.llm.summarizer.model == "gpt-4o"
        assert config.llm.summarizer.max_text_chars == 30000
        assert config.llm.summarizer.cost_control.max_cost_per_run == 2.0

    def test_independent_filter_and_summarizer(self, tmp_path):
        """Filter and summarizer have independent configs."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "llm": {
                "filter": {
                    "enabled": True,
                    "model": "gpt-4o-mini",
                    "cost_control": {
                        "max_cost_per_run": 0.10,
                    },
                },
                "summarizer": {
                    "enabled": True,
                    "model": "gpt-5-nano-2025-08-07",
                    "cost_control": {
                        "max_cost_per_run": 1.50,
                    },
                },
            },
        })
        config = load_config(path)
        assert config.llm.filter.enabled is True
        assert config.llm.summarizer.enabled is True
        assert config.llm.filter.model == "gpt-4o-mini"
        assert config.llm.summarizer.model == "gpt-5-nano-2025-08-07"
        assert config.llm.filter.cost_control.max_cost_per_run == 0.10
        assert config.llm.summarizer.cost_control.max_cost_per_run == 1.50


class TestWebConfig:
    def test_web_public_url_default_none(self, tmp_path):
        """public_url defaults to None when not set."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            }
        })
        config = load_config(path)
        assert config.web.public_url is None

    def test_web_public_url_loaded(self, tmp_path):
        """public_url is loaded from config."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "web": {
                "public_url": "https://digest.example.com",
            },
        })
        config = load_config(path)
        assert config.web.public_url == "https://digest.example.com"

    def test_web_public_url_strips_trailing_slash(self, tmp_path):
        """public_url has trailing slash stripped."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "web": {
                "public_url": "https://digest.example.com/",
            },
        })
        config = load_config(path)
        assert config.web.public_url == "https://digest.example.com"

    def test_web_public_url_adds_scheme_when_missing(self, tmp_path):
        """public_url without scheme gets http:// prepended."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "web": {
                "public_url": "127.0.0.1:8000",
            },
        })
        config = load_config(path)
        assert config.web.public_url == "http://127.0.0.1:8000"


class TestScoringWithoutAlpha:
    def test_scoring_has_no_alpha(self, tmp_path):
        """ScoringConfig no longer has alpha field."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            }
        })
        config = load_config(path)
        assert not hasattr(config.scoring, "alpha")
        assert not hasattr(config.scoring, "relevance")

    def test_scoring_has_quality_and_venue_tiers(self, tmp_path):
        """ScoringConfig still has quality weights and venue_tiers."""
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "scoring": {
                "quality": {
                    "w_venue": 0.35,
                    "w_code": 0.30,
                    "w_fresh": 0.35,
                },
                "venue_tiers": {
                    "tier1": ["CVPR", "ICCV"],
                    "tier2": ["ECCV", "ICRA"],
                },
            },
        })
        config = load_config(path)
        assert config.scoring.quality.w_venue == 0.35
        assert config.scoring.venue_tiers == {
            "tier1": ["CVPR", "ICCV"],
            "tier2": ["ECCV", "ICRA"],
        }


class TestExtraInstructions:
    def test_default_extra_instructions_none(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(MINIMAL_CONFIG)
        config = load_config(cfg_file)
        assert config.llm.filter.extra_instructions is None
        assert config.llm.summarizer.extra_instructions is None

    def test_filter_extra_instructions(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(MINIMAL_CONFIG + '\nllm:\n  filter:\n    extra_instructions: "Focus on robotics"\n')
        config = load_config(cfg_file)
        assert config.llm.filter.extra_instructions == "Focus on robotics"

    def test_summarizer_extra_instructions(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(MINIMAL_CONFIG + '\nllm:\n  summarizer:\n    extra_instructions: "Include benchmark results in one_liner"\n')
        config = load_config(cfg_file)
        assert config.llm.summarizer.extra_instructions == "Include benchmark results in one_liner"


class TestEnrichmentConfig:
    def test_enrichment_config_exists(self, tmp_path):
        """EnrichmentConfig is present on loaded config."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(MINIMAL_CONFIG)
        config = load_config(cfg_file)
        assert config.enrichment is not None
