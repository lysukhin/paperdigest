# === Builder ===
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir ".[llm,web]"

# === Runtime ===
FROM python:3.11-slim

# supercronic for container-friendly cron + tini as PID 1 (auto-detect arch)
ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tini \
    && SUPERCRONIC_ARCH=$(case "$TARGETARCH" in arm64) echo "linux-arm64" ;; *) echo "linux-amd64" ;; esac) \
    && curl -fsSL "https://github.com/aptible/supercronic/releases/download/v0.2.33/supercronic-${SUPERCRONIC_ARCH}" \
       -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["tini", "--"]

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
