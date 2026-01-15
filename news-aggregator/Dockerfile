# Build stage with models
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.109.0 \
    uvicorn>=0.27.0 \
    aiosqlite>=0.19.0 \
    feedparser>=6.0.10 \
    spacy>=3.7.0 \
    sentence-transformers>=2.2.0 \
    numpy>=1.24.0 \
    apscheduler>=3.10.4 \
    slowapi>=0.1.9

# Download spaCy model (~50MB)
RUN python -m spacy download en_core_web_sm

# Pre-download sentence transformer model (~80MB)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

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

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
