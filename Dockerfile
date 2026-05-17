FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev \
    libxslt-dev \
    libssl-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Download NLTK data (required by newspaper4k) ──────────────────────────────
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# ── Install Playwright Chromium browser (JS-heavy blog fallback) ───────────────
RUN playwright install chromium \
    && playwright install-deps chromium

# ── Copy project source ────────────────────────────────────────────────────────
COPY . .

# ── Create output directory ────────────────────────────────────────────────────
RUN mkdir -p output

# ── Environment defaults (override via docker-compose.yml or --env-file) ───────
ENV NCBI_EMAIL=researcher@example.com \
    RECENCY_LAMBDA=0.005 \
    CITATION_COUNT_MAX=50000 \
    TAGGING_MODEL=cross-encoder/nli-MiniLM2-L6-H4 \
    OUTPUT_PATH=output/scraped_data.json \
    PYTHONUNBUFFERED=1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "main.py"]
