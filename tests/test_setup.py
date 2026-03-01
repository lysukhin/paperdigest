"""Tests for setup wizard generation functions."""

from pathlib import Path

import yaml


from paperdigest.setup import (
    _is_ip_address,
    generate_caddyfile,
    generate_config,
    generate_crontab,
    generate_env,
)


class TestSetupConfigGeneration:
    def test_generate_config_default_topic(self, tmp_path):
        """Generated config has correct YAML structure, topic fields, and language."""
        path = tmp_path / "config.yaml"
        generate_config(
            path,
            topic_name="Autonomous Driving",
            topic_description="Papers about self-driving cars",
            primary_keywords=["autonomous driving", "self-driving"],
            arxiv_categories=["cs.CV", "cs.RO"],
            language="English",
            domain=None,
        )

        assert path.exists()
        data = yaml.safe_load(path.read_text())

        # Topic section
        assert data["topic"]["name"] == "Autonomous Driving"
        assert data["topic"]["description"] == "Papers about self-driving cars"
        assert data["topic"]["primary_keywords"] == ["autonomous driving", "self-driving"]
        assert data["topic"]["arxiv_categories"] == ["cs.CV", "cs.RO"]

        # LLM config — both filter and summarizer enabled with gpt-4o-mini
        assert data["llm"]["filter"]["enabled"] is True
        assert data["llm"]["filter"]["model"] == "gpt-4o-mini"
        assert data["llm"]["summarizer"]["enabled"] is True
        assert data["llm"]["summarizer"]["model"] == "gpt-4o-mini"

        # Language
        assert data["llm"]["summarizer"]["language"] == "English"

        # Web host always 0.0.0.0 for Docker
        assert data["web"]["host"] == "0.0.0.0"

        # No public_url when domain is None
        assert "public_url" not in data.get("web", {}) or data["web"].get("public_url") is None

    def test_generate_config_with_domain(self, tmp_path):
        """When domain is provided, web.public_url is set and host is 0.0.0.0."""
        path = tmp_path / "config.yaml"
        generate_config(
            path,
            topic_name="ML Research",
            topic_description="Machine learning papers",
            primary_keywords=["machine learning"],
            arxiv_categories=["cs.LG"],
            language="Russian",
            domain="digest.example.com",
        )

        data = yaml.safe_load(path.read_text())

        assert data["web"]["public_url"] == "https://digest.example.com:38443"
        assert data["web"]["host"] == "0.0.0.0"


class TestSetupEnvGeneration:
    def test_generate_env(self, tmp_path):
        """All keys present when values are provided."""
        path = tmp_path / ".env"
        generate_env(
            path,
            llm_api_key="sk-test-key",
            openai_admin_key="sk-admin-key",
            telegram_bot_token="123:ABC",
            telegram_chat_id="84555880",
        )

        assert path.exists()
        content = path.read_text()
        assert "LLM_API_KEY=sk-test-key" in content
        assert "OPENAI_ADMIN_KEY=sk-admin-key" in content
        assert "TELEGRAM_BOT_TOKEN=123:ABC" in content
        assert "TELEGRAM_CHAT_ID=84555880" in content

    def test_generate_env_no_telegram(self, tmp_path):
        """Empty telegram values are omitted from the .env file."""
        path = tmp_path / ".env"
        generate_env(
            path,
            llm_api_key="sk-test-key",
            openai_admin_key="",
            telegram_bot_token="",
            telegram_chat_id="",
        )

        content = path.read_text()
        assert "LLM_API_KEY=sk-test-key" in content
        assert "TELEGRAM_BOT_TOKEN" not in content
        assert "TELEGRAM_CHAT_ID" not in content
        assert "OPENAI_ADMIN_KEY" not in content

    def test_generate_env_empty(self, tmp_path):
        """Empty file when all values are empty strings."""
        path = tmp_path / ".env"
        generate_env(
            path,
            llm_api_key="",
            openai_admin_key="",
            telegram_bot_token="",
            telegram_chat_id="",
        )

        content = path.read_text()
        # Should have no KEY=value lines (may have empty string or nothing)
        lines = [line for line in content.splitlines() if line.strip() and not line.startswith("#")]
        assert len(lines) == 0


class TestSetupCaddyfile:
    def test_generate_caddyfile_with_domain(self, tmp_path):
        """Caddyfile uses domain as host block when provided."""
        path = tmp_path / "Caddyfile"
        generate_caddyfile(path, domain="digest.example.com")

        content = path.read_text()
        assert "digest.example.com" in content
        assert "reverse_proxy web:8000" in content

    def test_generate_caddyfile_no_domain(self, tmp_path):
        """Caddyfile uses :80 as fallback when no domain provided."""
        path = tmp_path / "Caddyfile"
        generate_caddyfile(path, domain=None)

        content = path.read_text()
        assert ":80" in content
        assert "reverse_proxy web:8000" in content

    def test_generate_caddyfile_ip_address(self, tmp_path):
        """Caddyfile uses :80 when domain is an IP address (no TLS)."""
        path = tmp_path / "Caddyfile"
        generate_caddyfile(path, domain="192.168.1.100")

        content = path.read_text()
        assert ":80" in content
        assert "192.168.1.100" not in content
        assert "reverse_proxy web:8000" in content


class TestIPAddressDetection:
    def test_ipv4(self):
        assert _is_ip_address("192.168.1.1") is True
        assert _is_ip_address("10.0.0.1") is True
        assert _is_ip_address("31.97.65.162") is True

    def test_ipv6(self):
        assert _is_ip_address("::1") is True
        assert _is_ip_address("2001:db8::1") is True

    def test_fqdn(self):
        assert _is_ip_address("digest.example.com") is False
        assert _is_ip_address("my-server.local") is False


class TestSetupCrontab:
    def test_generate_crontab_default(self, tmp_path):
        """Crontab contains the schedule and paperdigest run command."""
        path = tmp_path / "crontab"
        generate_crontab(path, schedule="0 8 * * 1")

        content = path.read_text()
        assert "0 8 * * 1" in content
        assert "/opt/venv/bin/paperdigest run --config /app/config.yaml" in content
        assert ">> /app/data/cron.log 2>&1" in content

    def test_generate_crontab_custom(self, tmp_path):
        """Crontab respects a custom schedule."""
        path = tmp_path / "crontab"
        generate_crontab(path, schedule="30 14 * * *")

        content = path.read_text()
        assert "30 14 * * *" in content
        assert "/opt/venv/bin/paperdigest run --config /app/config.yaml" in content
