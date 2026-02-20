"""Tests for config loading."""

from pathlib import Path

import pytest
import yaml

from paperdigest.config import load_config


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
        assert config.scoring.alpha == 0.65  # default

    def test_missing_topic_raises(self, tmp_path):
        path = _write_config(tmp_path, {"scoring": {"alpha": 0.5}})
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
                "alpha": 0.8,
                "quality": {
                    "w_venue": 0.30,
                    "w_author": 0.20,
                    "w_cite": 0.20,
                    "w_code": 0.10,
                    "w_fresh": 0.20,
                },
            },
        })
        config = load_config(path)
        assert config.scoring.alpha == 0.8
        assert config.scoring.quality.w_venue == 0.30

    def test_llm_defaults_disabled(self, tmp_path):
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
        })
        config = load_config(path)
        assert config.llm.enabled is False

    def test_llm_cost_control(self, tmp_path):
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "llm": {
                "enabled": True,
                "model": "gpt-4o-mini",
                "cost_control": {
                    "max_cost_per_run": 1.0,
                    "max_cost_per_month": 20.0,
                },
            },
        })
        config = load_config(path)
        assert config.llm.enabled is True
        assert config.llm.cost_control.max_cost_per_run == 1.0
        assert config.llm.cost_control.max_cost_per_month == 20.0

    def test_invalid_alpha_raises(self, tmp_path):
        path = _write_config(tmp_path, {
            "topic": {
                "name": "Test",
                "primary_keywords": ["test"],
            },
            "scoring": {"alpha": 1.5},
        })
        with pytest.raises(ValueError, match="alpha"):
            load_config(path)

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
