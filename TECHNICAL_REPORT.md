# Technical Report: Async Multi-Source Data Scraping Pipeline & Trust Scoring Engine

**Author:** Staff Software Engineer / Core Data Architect
**Date:** 2026-05-16
**Scope:** Engineering take-home assignment — data pipeline design, algorithmic trust scoring, NLP tagging, and abuse prevention

---

## 1. Data Collection Pipeline Design

### 1.1 Architecture Overview

The pipeline is structured as an asynchronous Python application using `asyncio.gather()` to run all scraper groups concurrently. Each scraper module inherits from `BaseScraper`, which provides shared infrastructure: session management, User-Agent/Referer rotation, and exponential backoff retry via `tenacity`.

Three source-specific scrapers handle heterogeneous content:

**Blog Scraper (`newspaper4k` + Playwright)**
The primary extraction path uses `newspaper4k`, an actively maintained fork of `newspaper3k`, which strips boilerplate navigation and advertisements using CSS heuristics and NLP-guided text density scoring. When the extracted text is fewer than 200 characters — indicating a JavaScript-rendered SPA — a `Playwright` Chromium headless browser launches asynchronously, navigates the page, waits for network idle, and extracts inner text from `<article>`, `<main>`, or `<body>` in that priority order. This two-stage design avoids the overhead of browser automation for the majority of static-HTML blogs.

**YouTube Scraper (`yt-dlp` + `youtube-transcript-api`)**
Metadata (channel name, upload date, description, view/like counts, tags) is extracted via `yt-dlp` using `skip_download=True`. This bypasses the YouTube Data API entirely, eliminating quota management complexity. Transcripts are retrieved via `youtube-transcript-api`, which fetches caption XML directly from YouTube's timedtext endpoint. If manual or auto-generated English captions are unavailable (`TranscriptsDisabled` / `NoTranscriptFound`), the video description serves as the content body for downstream NLP.

**PubMed Scraper (Biopython Entrez)**
The scraper interfaces exclusively with NCBI's Entrez XML API (`efetch` endpoint) — never the HTML frontend. This ensures structured, machine-readable output regardless of frontend changes. Article metadata (title, author list, journal, abstract, publication date, DOI) is parsed from the returned XML tree using Python's standard `xml.etree.ElementTree`. Citation count is retrieved separately via `Entrez.elink` querying the PMC link database.

### 1.2 Resilience Design

- **Retry logic**: `tenacity` decorates the inner fetch loop with up to 3 attempts, exponential backoff (1s → 2s → 4s), retrying on `aiohttp.ClientError`, `asyncio.TimeoutError`, and HTTP 429/503 responses.
- **Graceful degradation**: All scrapers return a valid (possibly sparse) dictionary on failure. `asyncio.gather(return_exceptions=True)` prevents one failing source from blocking the rest.
- **Missing field defaults**: Empty authors default to `"Unknown"`; missing dates apply an `R=0.30` recency penalty; empty text produces zero content chunks.

---

## 2. Zero-Shot Tagging Design

### 2.1 Model Selection

Zero-shot classification is performed using a Natural Language Inference (NLI) model from HuggingFace Transformers. The model framing is: *"Does this text entail the hypothesis 'This text is about {label}'?"*

The default model, `cross-encoder/nli-MiniLM2-L6-H4` (~80 MB), is a distilled cross-encoder fine-tuned on MNLI. It offers strong accuracy on domain classification tasks at a fraction of the cost of `facebook/bart-large-mnli` (~1.6 GB). The model runs CPU-side, making it viable in any environment without GPU requirements. Users needing higher accuracy can set `TAGGING_MODEL=facebook/bart-large-mnli` in `.env`.

### 2.2 Label Design and Confidence Filtering

The candidate label set (`["Artificial Intelligence", "Healthcare", "Data Science", ...]`) covers the primary topical domains expected from the target sources. `multi_label=True` is passed to the pipeline, enabling multiple labels to be returned. Only labels with entailment probability > 0.20 are included in `topic_tags`, with a soft cap of 5 labels per item. This threshold was chosen empirically to balance recall (catching relevant secondary topics) against precision (avoiding noise labels).

### 2.3 Subjectivity Analysis (Abuse Prevention)

TextBlob's `sentiment.subjectivity` is applied to the first 2,000 characters of each article. It uses a pre-built sentiment lexicon mapping individual words and phrases to polarity and subjectivity values. A score above 0.60 flags the content as opinion-heavy or emotionally loaded — a strong signal of SEO-bait, low-quality content farms, or bias. This triggers a `0.5×` multiplier on `A_cred` in the scoring engine.

---

## 3. Formal Trust Score Algorithm

### 3.1 Formula

$$\text{Trust Score} = w_1 \cdot A_{\text{cred}} + w_2 \cdot C_{\text{score}} + w_3 \cdot D_{\text{auth}} + w_4 \cdot R + w_5 \cdot M$$

All five sub-scores are explicitly normalised to $[0, 1]$. Weights $\sum w_i = 1.0$ vary by source type.

### 3.2 Grounding in Pre-Established Algorithms

**TrustRank (Gyöngyi et al., 2004)**
> "Combating Web Spam with TrustRank," WWW 2004.

TrustRank propagates trust outward from manually verified "seed" pages, penalising domains that are distant from trusted anchors. This pipeline operationalises TrustRank through `D_auth`: domains matching a curated whitelist (major AI labs, `.edu`, `.gov`, well-known publishers) receive scores of 0.8–1.0. Domains on the blacklist (spam farms, abusive TLDs like `.xyz`, `.tk`) receive a hard score of 0.0, blocking all trust propagation from tainted sources. This is structurally analogous to TrustRank's "spam prevention" property — no matter how high a blacklisted domain scores on other sub-components, the final score floors at zero.

**PageRank (Page et al., 1999)**
> "The PageRank Citation Ranking: Bringing Order to the Web."

PageRank models a node's authority as proportional to the sum of the authority of all nodes linking to it. For academic content, citations serve as directed authority links. A paper cited by 5,000 papers has greater epistemic authority than one cited by 5. The pipeline log-normalises citation counts as:

$$C_{\text{score}} = \frac{\ln(1 + C)}{\ln(1 + C_{\max})}$$

This mirrors PageRank's resistance to outlier dominance: highly-cited papers (e.g., ImageNet with 100,000+ citations) don't produce a score of 100× a paper with 1,000 citations — the log scale compresses the distribution to $[0, 1]$.

**Google E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)**
> Google Search Quality Evaluator Guidelines, 2023 revision.

E-E-A-T introduces the "Your Money or Your Life" (YMYL) content category — pages covering health, medicine, finance, or law that can directly harm readers if inaccurate. The pipeline implements YMYL detection by checking `topic_tags` for medical keywords (`healthcare`, `clinical`, `disease`, etc.) and whether the `source_type` is `pubmed`. If YMYL content is identified but no medical disclaimer string is found in the article body, the modifier $M$ drops to $0.25$ — a 75% penalty on that weight component. This strongly discourages medical misinformation from achieving a high trust score. PubMed articles, while inherently medical, typically contain scientific language rather than patient-facing disclaimers; the penalty acknowledges this but still rewards explicit safety statements.

**Recency Decay**
Adapted from information half-life models (Sanderson & Croft, 1999; used in web freshness ranking literature):

$$R = e^{-\lambda \cdot t}$$

where $t$ is days since publication and $\lambda = 0.005$ by default. This gives an approximately 2-year half-life for content relevance.

---

## 4. Systemic Abuse Defenses

### 4.1 Domain Blacklist (TrustRank Hard Block)
A static `frozenset` of known spam domains and low-reputation TLDs (`.xyz`, `.tk`, `.ml`, `.pw`, `.buzz`) is checked at the entry point of every HTTP request in `BaseScraper._validate_domain()`. A `BlockedDomainError` is raised immediately — no HTTP request is made, and `trust_score = 0.0` is enforced. This prevents the pipeline from being weaponised to launder spam content through a trusted-appearing output file.

### 4.2 NLP Subjectivity Multiplier (SEO Spam Detection)
Content with a TextBlob subjectivity score > 0.60 has its `A_cred` halved. Highly emotional or opinionated writing is a reliable signal of low-quality content, SEO link-bait, or astroturfing campaigns. This penalty is logged with the specific score for auditability.

### 4.3 Missing Metadata Penalties
Rather than failing silently, each missing field applies an explicit mathematical penalty and emits a structured warning:
- Missing author → `A_cred = 0.0`
- Missing date → `R = 0.30` (fixed, below the one-year equivalent ~0.16)
- Missing content → `content_chunks = []` (flagged by Pydantic `model_validator`)

### 4.4 Multiple Authors
When a source has multiple authors (common in PubMed papers), the display field shows `"First Author et al. (+N)"` while the full `authors_list` is retained internally. The `raw_author_credibility` signal is computed from the source-level engagement or publication record — not averaged across authors — because the credibility of multi-author academic papers is best captured by the venue (journal/conference) rather than individual author H-indices, which are not readily available without additional API calls.

### 4.5 User-Agent and Referer Rotation
Every HTTP request through `BaseScraper._fetch()` samples a fresh random User-Agent string from `fake-useragent`'s browser pool and a random Referer from a rotation pool mimicking organic Google/Bing/DuckDuckGo traffic. Combined with exponential backoff on 429 responses, this significantly reduces the likelihood of IP-level rate limiting blocking the pipeline.

---

*End of Technical Report*
