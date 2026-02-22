# === Builder ===
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir ".[llm,web]"

# === Runtime ===
FROM python:3.11-slim

# supercronic for container-friendly cron
ARG SUPERCRONIC_URL=https://github.com/aptible/supercronic/releases/download/v0.2.33/supercronic-linux-amd64
ARG SUPERCRONIC_SHA=71b0d58cc53f6bd72cf2f293e09e294b67c30571
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -fsSL "$SUPERCRONIC_URL" -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY config.yaml.example /app/config.yaml.example

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    NO_PROGRESS=1

# Default data directory
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["paperdigest", "serve", "--config", "/app/config.yaml"]
