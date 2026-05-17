from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from newspaper import Article
from newspaper.article import ArticleException

from scraper.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TrustRank Seed Whitelist — High-authority domains
# ---------------------------------------------------------------------------
# Mirrors the TrustRank principle: we start from a set of manually verified
# trusted nodes; proximity to these anchors raises a domain's authority score.
TRUSTED_BLOG_DOMAINS: frozenset[str] = frozenset(
    {
        # Research / academia
        "distill.pub",
        "karpathy.github.io",
        "lilianweng.github.io",
        "ruder.io",
        "colah.github.io",
        "gregorygundersen.com",
        # Major AI labs
        "openai.com",
        "deepmind.google",
        "ai.googleblog.com",
        "research.facebook.com",
        "ai.meta.com",
        "huggingface.co",
        "blogs.microsoft.com",
        # Developer / tech media
        "towardsdatascience.com",
        "medium.com",
        "techcrunch.com",
        "wired.com",
        "theverge.com",
        "arstechnica.com",
        "simonwillison.net",
        "martinfowler.com",
        # ML frameworks
        "pytorch.org",
        "tensorflow.org",
    }
)


class BlogScraper(BaseScraper):
    """
    Returns a dict with keys consumed by ``ScrapedItem`` (Pydantic model):
      source_url, source_type, title, author, published_date,
      text, domain, domain_authority.
    """

    async def scrape(self, url: str) -> dict[str, Any]:
        """
        Scrape a blog post at ``url``.
        Tries newspaper4k first; falls back to Playwright on thin content.
        """
        logger.info("[BlogScraper] Scraping: %s", url)
        self._validate_domain(url)

        result = await self._scrape_newspaper(url)

        # Fallback: thin content indicates JS-rendered page
        if len(result.get("text", "")) < 200:
            logger.warning(
                "[BlogScraper] Thin content (%d chars) from newspaper4k — "
                "activating Playwright fallback for %s",
                len(result.get("text", "")),
                url,
            )
            result = await self._scrape_playwright(url, result)

        # Attach domain authority (TrustRank score)
        domain = result.get("domain", "")
        result["domain_authority"] = self.compute_domain_authority(domain)

        return result

    # ------------------------------------------------------------------
    # Stage 1: newspaper4k
    # ------------------------------------------------------------------
    async def _scrape_newspaper(self, url: str) -> dict[str, Any]:
        """Primary extraction using newspaper4k (sync under async wrapper)."""
        result: dict[str, Any] = {
            "source_url": url,
            "source_type": "blog",
            "title": None,
            "author": None,
            "published_date": None,
            "text": "",
            "top_image": None,
            "keywords": [],
            "domain": urlparse(url).netloc.lower().lstrip("www."),
        }

        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()

            result["title"] = article.title or None
            result["text"] = article.text or ""
            result["top_image"] = str(article.top_image) if article.top_image else None
            result["keywords"] = list(article.keywords or [])

            # Author handling — newspaper4k returns a list
            authors: list[str] = list(article.authors or [])
            if authors:
                result["author"] = ", ".join(authors)
            else:
                logger.warning(
                    "[BlogScraper] No author found for %s — "
                    "A_cred penalty will apply.",
                    url,
                )

            # Published date
            if article.publish_date:
                result["published_date"] = article.publish_date.isoformat()
            else:
                logger.warning(
                    "[BlogScraper] No publish_date for %s — "
                    "R (recency) penalty will apply.",
                    url,
                )

        except ArticleException as exc:
            logger.error("[BlogScraper] newspaper4k failed for %s: %s", url, exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("[BlogScraper] Unexpected error for %s: %s", url, exc)

        return result
    async def _scrape_playwright(
        self, url: str, partial: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            from playwright.async_api import async_playwright  # lazy import

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=self._ua.random,
                    extra_http_headers={"Referer": "https://www.google.com/"},
                )
                page = await context.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30_000)

                for selector in ("article", "main", "[role='main']", "body"):
                    element = await page.query_selector(selector)
                    if element:
                        text = await element.inner_text()
                        if len(text.strip()) > 200:
                            partial["text"] = text.strip()
                            logger.info(
                                "[BlogScraper] Playwright extracted %d chars "
                                "via selector '%s' for %s",
                                len(partial["text"]),
                                selector,
                                url,
                            )
                            break

                # Best-effort title extraction if newspaper4k missed it
                if not partial.get("title"):
                    h1 = await page.query_selector("h1")
                    if h1:
                        partial["title"] = (await h1.inner_text()).strip()

                await browser.close()

        except ImportError:
            logger.error(
                "[BlogScraper] Playwright not installed. "
                "Run: playwright install chromium"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[BlogScraper] Playwright fallback failed for %s: %s", url, exc)

        return partial
    @staticmethod
    def compute_domain_authority(domain: str) -> float:
        clean = domain.lower().lstrip("www.")

        if clean in TRUSTED_BLOG_DOMAINS:
            return 1.0

        # Subdomain of a trusted seed (e.g., blog.huggingface.co)
        if any(clean.endswith("." + td) for td in TRUSTED_BLOG_DOMAINS):
            return 0.8

        # Academic / government TLD — inherently authoritative
        if clean.endswith(".edu") or clean.endswith(".gov"):
            return 0.9

        return 0.5  # Neutral — unknown provenance
