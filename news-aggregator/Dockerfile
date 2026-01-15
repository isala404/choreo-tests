# Build stage with models
FROM python:3.12.8-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies with pinned versions
RUN pip install --no-cache-dir \
    fastapi==0.115.7 \
    uvicorn==0.34.0 \
    aiosqlite==0.20.0 \
    feedparser==6.0.11 \
    spacy==3.8.3 \
    sentence-transformers==3.3.1 \
    numpy==2.2.2 \
    apscheduler==3.11.0 \
    slowapi==0.1.9 \
    httpx==0.28.1

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-download sentence transformer model to cache
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Runtime stage
FROM python:3.12.8-slim-bookworm

WORKDIR /app

# Install security updates
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model caches from builder
COPY --from=builder /app/.cache /app/.cache

# Create non-root user with UID 10001
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -d /app -s /sbin/nologin appuser

# Create data directory
RUN mkdir -p /app/data && chown -R appuser:appgroup /app

# Copy application
COPY main.py ./

# Set ownership
RUN chown -R appuser:appgroup /app

USER 10001

EXPOSE 8000

ENV PORT=8000
ENV DATABASE_PATH=/app/data/news.db
ENV FETCH_INTERVAL=5
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
