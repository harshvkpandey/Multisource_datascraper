
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from models import ScrapedItem
from scraper.base_scraper import BlockedDomainError
from scraper.blog_scraper import BlogScraper
from scraper.pubmed_scraper import PubMedScraper
from scraper.youtube_scraper import YouTubeScraper
from scoring.trust_score import compute_trust_score
from utils.chunking import RecursiveChunker
from utils.tagging import ZeroShotTagger
# Target sources
BLOG_URLS: list[str] = [
    "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
    "https://ruder.io/optimizing-gradient-descent/",
    "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/",
]
YOUTUBE_URLS: list[str] = [
    "https://www.youtube.com/watch?v=aircAruvnKk",   # 3Blue1Brown: Neural Networks
    "https://www.youtube.com/watch?v=kCc8FmEb1nY",   # Karpathy: Let's Build GPT
]
PUBMED_URLS: list[str] = [
    "https://pubmed.ncbi.nlm.nih.gov/33423054/",     # AI in genomic diagnostics
]

OUTPUT_PATH: Path = Path(os.getenv("OUTPUT_PATH", "output/scraped_data.json"))
def detect_language(text: str) -> str:
    """Detect ISO 639-1 language code from text. Returns 'en' on failure."""
    if not text or not text.strip():
        return "en"
    try:
        from langdetect import detect
        return detect(text[:1000])
    except Exception:  # noqa: BLE001
        return "en"
def build_item(
    raw: dict[str, Any],
    tagger: ZeroShotTagger,
    chunker: RecursiveChunker,
) -> ScrapedItem:
    text: str = raw.get("text") or raw.get("abstract") or raw.get("description") or ""
    source_type = raw.get("source_type", "blog")

    # Step 1: Language detection
    language = detect_language(text)

    # Step 2: Zero-shot topic tagging
    topic_tags = tagger.get_tags(text)

    # Step 3: Subjectivity (E-E-A-T NLP abuse check)
    subjectivity = ZeroShotTagger.get_subjectivity(text)

    # Step 4: Trust score
    trust_score = compute_trust_score(
        source_type=source_type,
        author=raw.get("author"),
        published_date=raw.get("published_date"),
        domain_authority=raw.get("domain_authority", 0.5),
        citation_count=raw.get("citation_count", 0),
        raw_author_credibility=raw.get("author_credibility_proxy", 0.5),
        subjectivity_score=subjectivity,
        topic_tags=topic_tags,
        content_text=text,
        is_blacklisted=False,  # BlockedDomainError is raised before reaching here
    )

    # Step 5: Chunk content
    chunks = chunker.split(text)

    # Step 6: Validate
    return ScrapedItem(
        source_url=raw["source_url"],
        source_type=source_type,
        author=raw.get("author") or "Unknown",
        published_date=raw.get("published_date"),
        language=language,
        region=raw.get("region"),
        topic_tags=topic_tags,
        trust_score=trust_score,
        content_chunks=chunks,
    )

async def scrape_blogs() -> list[dict[str, Any]]:
    """Scrape all configured blog URLs concurrently."""
    async with BlogScraper() as scraper:
        tasks = [scraper.scrape(url) for url in BLOG_URLS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    items: list[dict[str, Any]] = []
    for url, res in zip(BLOG_URLS, results):
        if isinstance(res, Exception):
            logger.error("[main] Blog scrape failed for %s: %s", url, res)
        else:
            items.append(res)
    return items


async def scrape_youtube() -> list[dict[str, Any]]:
    """Scrape all configured YouTube URLs concurrently."""
    async with YouTubeScraper() as scraper:
        tasks = [scraper.scrape(url) for url in YOUTUBE_URLS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    items: list[dict[str, Any]] = []
    for url, res in zip(YOUTUBE_URLS, results):
        if isinstance(res, Exception):
            logger.error("[main] YouTube scrape failed for %s: %s", url, res)
        else:
            items.append(res)
    return items


async def scrape_pubmed() -> list[dict[str, Any]]:
    """Scrape all configured PubMed URLs (synchronous Entrez calls, small list)."""
    async with PubMedScraper() as scraper:
        tasks = [scraper.scrape(url) for url in PUBMED_URLS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    items: list[dict[str, Any]] = []
    for url, res in zip(PUBMED_URLS, results):
        if isinstance(res, Exception):
            logger.error("[main] PubMed scrape failed for %s: %s", url, res)
        else:
            items.append(res)
    return items

async def run_pipeline() -> None:
    """
    Execute the full scraping pipeline:
      1. Run all scrapers concurrently.
      2. Process, tag, score, and chunk each item.
      3. Validate via Pydantic.
      4. Write JSON output.
    """
    logger.info("=" * 60)
    logger.info("Pipeline starting — scraping %d sources", 6)
    logger.info("=" * 60)

    # Initialise shared utilities once
    tagger = ZeroShotTagger()
    chunker = RecursiveChunker()

    # Run all three scraper groups concurrently
    blog_raws, yt_raws, pm_raws = await asyncio.gather(
        scrape_blogs(),
        scrape_youtube(),
        scrape_pubmed(),
    )

    all_raws = blog_raws + yt_raws + pm_raws
    logger.info("Scraping complete. Processing %d items…", len(all_raws))

    validated_items: list[ScrapedItem] = []
    for raw in all_raws:
        try:
            item = build_item(raw, tagger, chunker)
            validated_items.append(item)
            logger.info(
                "[main] ✓ %s | trust=%.4f | tags=%s",
                raw.get("source_url", "?"),
                item.trust_score,
                item.topic_tags,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[main] Validation failed for %s: %s",
                raw.get("source_url", "?"),
                exc,
            )

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.model_dump(mode="json") for item in validated_items]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(
        "Pipeline complete. %d/%d items written to %s",
        len(validated_items),
        len(all_raws),
        OUTPUT_PATH,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_pipeline())
