# ══════════════════════════════════════════════════════════════════
#  Multi-stage Dockerfile for the News Trading System
#  Python: 3.11-slim (Railway compatible)
# ══════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
WORKDIR /app
COPY . .
RUN mkdir -p logs && chmod +x entrypoint.sh

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Railway sets $PORT dynamically — do not hardcode or use Docker HEALTHCHECK
# (Railway handles healthchecks externally via railway.toml healthcheckPath)
EXPOSE ${PORT:-8501}

CMD ["sh", "/app/entrypoint.sh"]
