# Async Multi-Source Data Scraping Pipeline & Trust Scoring Engine

A production-ready, asynchronous Python pipeline that scrapes heterogeneous content sources (blogs, YouTube, PubMed), applies automated NLP processing, and computes a mathematically grounded trust score for each item.

---

## Architecture

```
project/
├── scraper/
│   ├── base_scraper.py     # Abstract base: aiohttp, UA rotation, retry, blacklist
│   ├── blog_scraper.py     # newspaper4k primary + Playwright JS fallback
│   ├── youtube_scraper.py  # yt-dlp metadata + youtube-transcript-api transcripts
│   └── pubmed_scraper.py   # NCBI Entrez efetch XML + elink citation count
│
├── scoring/
│   └── trust_score.py      # Weighted mathematical trust scoring engine
│
├── utils/
│   ├── chunking.py         # Recursive character text chunker (512 chars, 10% overlap)
│   └── tagging.py          # Zero-shot NLI topic tagger + TextBlob subjectivity
│
├── models.py               # Pydantic V2 ScrapedItem validation schema
├── main.py                 # Async orchestrator (asyncio.gather across all scrapers)
│
├── tests/
│   └── test_scoring.py     # Pytest: 30+ tests on math invariants & edge cases
│
├── Dockerfile
├── docker-compose.yml
└── output/scraped_data.json  (auto-generated)
```

---

## Library Choices

| Purpose | Library | Rationale |
|---|---|---|
| Async HTTP | `aiohttp` | Fastest async HTTP client; native connection pooling |
| Blog extraction | `newspaper4k` | Production-grade boilerplate stripping + NLP |
| JS-heavy pages | `playwright` | Full Chromium rendering; lazy-loaded only on fallback |
| YouTube metadata | `yt-dlp` | No API key; richer data than official API |
| YouTube transcripts | `youtube-transcript-api` | Zero-quota transcript access |
| PubMed API | `biopython` (Entrez) | Official NCBI wrapper; structured XML response |
| Zero-shot tagging | `transformers` (HuggingFace) | NLI-based; no training data required |
| Subjectivity | `textblob` | Lightweight lexicon; sufficient for abuse flag |
| Language detection | `langdetect` | Fast, accurate ISO 639-1 detection |
| Retry / backoff | `tenacity` | Production-grade decorator-based retry logic |
| UA rotation | `fake-useragent` | Realistic browser UA pool |
| Validation | `pydantic v2` | Strict schema enforcement with custom validators |
| Environment | `python-dotenv` | 12-factor app config pattern |

---

## Quick Start (Local)

```bash
# 1. Clone and enter directory
cd datascraping

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download Playwright browser
playwright install chromium

# 5. Configure environment
copy .env.example .env
# Edit .env: set NCBI_EMAIL=your_real_email@example.com

# 6. Run the pipeline
python main.py

# Output: output/scraped_data.json
```

---

## Docker (Recommended)

```bash
# Build and run (first run downloads ~80 MB NLI model)
docker-compose up --build

# Subsequent runs use cached model (named volume: huggingface_cache)
docker-compose up

# Output appears in ./output/scraped_data.json on your host machine
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NCBI_EMAIL` | `researcher@example.com` | **Required by NCBI** — set to your real email |
| `NCBI_API_KEY` | *(empty)* | Optional free key → 10 req/s (vs 3 req/s) |
| `YOUTUBE_API_KEY` | *(empty)* | Not required; yt-dlp works without it |
| `TAGGING_MODEL` | `cross-encoder/nli-MiniLM2-L6-H4` | Swap to `facebook/bart-large-mnli` for higher accuracy |
| `RECENCY_LAMBDA` | `0.005` | Decay rate for recency score |
| `CITATION_COUNT_MAX` | `50000` | Log-normalisation ceiling for citation score |
| `OUTPUT_PATH` | `output/scraped_data.json` | Output file path |

---

## Running Tests

```bash
pytest tests/ -v
```

Expected: **30+ passing tests** covering all mathematical invariants, edge cases, and schema validation.

---

## Trust Score Mathematical Breakdown

### Formula

```
Trust Score = w1·A_cred + w2·C_score + w3·D_auth + w4·R + w5·M
```

All sub-scores ∈ [0, 1]. Weights sum to 1.0 per source type.

### Dynamic Weight Tables

| Source | w1 A_cred | w2 C_score | w3 D_auth | w4 R | w5 M |
|--------|-----------|------------|-----------|------|------|
| blog | 0.25 | 0.05 | **0.35** | 0.20 | 0.15 |
| youtube | **0.30** | 0.05 | 0.25 | **0.25** | 0.15 |
| pubmed | 0.20 | **0.35** | 0.15 | 0.15 | 0.15 |

### Sub-score Definitions

**A_cred — Author Credibility**
- Missing author → `0.0` (hard zero + warning log)
- TextBlob subjectivity > 0.6 → `raw_score × 0.5` (E-E-A-T SEO flag)

**C_score — Citation Score (PageRank Proxy)**
```
C_score = ln(1 + C) / ln(1 + C_max)
```
Log-normalisation prevents outlier dominance. `C_max = 50,000`.

**D_auth — Domain Authority (TrustRank)**
```
1.0  Verified trusted seed domain (whitelist)
0.9  .edu / .gov TLD
0.8  Subdomain of trusted seed
0.5  Unknown / unverified domain
0.0  Blacklisted domain (hard zero, entire score zeroed)
```

**R — Recency (Exponential Decay)**
```
R = e^(-λ · t)    where t = days since publication, λ = 0.005
```
| Age | R score |
|-----|---------|
| Today (t=0) | 1.000 |
| 1 year (t=365) | 0.160 |
| 2 years (t=730) | 0.026 |
| Missing date | **0.30** (penalty) |

**M — Medical/YMYL Modifier (Google E-E-A-T)**
```
M = 1.0   non-medical content         (neutral)
M = 1.0   medical + disclaimer found  (safe)
M = 0.25  medical + no disclaimer     (severe YMYL penalty)
```
PubMed source type always triggers YMYL check.

### Abuse Prevention

| Condition | Effect |
|---|---|
| Blacklisted domain | `trust_score = 0.0` (hard zero) |
| Blacklisted TLD (`.xyz`, `.tk`, …) | `BlockedDomainError` raised |
| Missing author | `A_cred = 0.0` |
| High subjectivity (>0.6) | `A_cred × 0.5` |
| Missing publication date | `R = 0.30` |
| YMYL without disclaimer | `M = 0.25` |

---

## Output Schema (Pydantic V2)

```json
{
  "source_url": "https://...",
  "source_type": "blog | youtube | pubmed",
  "author": "string",
  "published_date": "ISO-8601 or null",
  "language": "en",
  "region": "US or null",
  "topic_tags": ["Artificial Intelligence", "Machine Learning"],
  "trust_score": 0.8742,
  "content_chunks": ["chunk 1...", "chunk 2..."]
}
```

---

## Structural Limitations

1. **Transcript availability**: YouTube auto-generated captions may be unavailable for some videos. The pipeline gracefully falls back to video descriptions.
2. **Rate limits**: NCBI Entrez limits unauthenticated callers to 3 req/s. Setting `NCBI_API_KEY` raises this to 10 req/s.
3. **newspaper4k accuracy**: Very heavily JavaScript-rendered sites may require the Playwright fallback, adding ~5-10 seconds per page.
4. **Citation freshness**: `elink` PMC citation counts reflect only PMC-indexed citations, not total cross-publisher citations (Google Scholar would give higher counts).
5. **NLI model**: `cross-encoder/nli-MiniLM2-L6-H4` is trained on NLI pairs, not a domain-specific topic classifier. Accuracy improves with `facebook/bart-large-mnli` but at ~20× the model size.
6. **Subjectivity proxy**: TextBlob's lexicon-based subjectivity is a heuristic. Transformer-based alternatives (e.g., `cardiffnlp/twitter-roberta-base-sentiment`) would be more accurate but significantly heavier.
