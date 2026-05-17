from __future__ import annotations
import abc
import asyncio
import logging
import random
from typing import Any
from urllib.parse import urlparse

import aiohttp
from fake_useragent import UserAgent
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
BLACKLISTED_DOMAINS: frozenset[str] = frozenset(
    {
        "spammy-farm.com",
        "content-mill.net",
        "articlefarm.biz",
        "linkfarming.info",
        "seo-spam.xyz",
        "clickbait.buzz",
        "fakemed.ru",
        "malware-news.tk",
        "pirated-content.ml",
        "casino-seo.pw",
        "autoblog-generator.net",
        "scraped-content.org",
    }
)

# Low-reputation TLDs frequently associated with spam/abuse
BLACKLISTED_TLDS: frozenset[str] = frozenset(
    {".xyz", ".tk", ".ml", ".pw", ".cf", ".gq", ".buzz", ".click", ".loan", ".download"}
)

# Rotation pool for Referer headers to mimic organic browser traffic
_REFERERS: list[str] = [
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://duckduckgo.com/",
    "https://www.reddit.com/",
    "https://scholar.google.com/",
    "https://www.twitter.com/",
]
class BlockedDomainError(Exception):
    """Raised when a URL's domain matches the TrustRank blacklist."""
class ScrapingError(Exception):
    """Generic scraping failure after all retries exhausted."""
class BaseScraper(abc.ABC):
    def __init__(self, timeout: int = 30, max_retries: int = 3) -> None:
        """
        Args:
            timeout: Total request timeout in seconds.
            max_retries: Maximum retry attempts before raising ScrapingError.
        """
        self._ua = UserAgent(fallback="Mozilla/5.0")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries
        self._session: aiohttp.ClientSession | None = None
    async def __aenter__(self) -> "BaseScraper":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self
    async def __aexit__(self, *_: Any) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    @staticmethod
    def _validate_domain(url: str) -> None:
        parsed = urlparse(url)
        host: str = parsed.netloc.lower().removeprefix("www.")

        if host in BLACKLISTED_DOMAINS:
            raise BlockedDomainError(
                f"Domain '{host}' is on the TrustRank blacklist. "
                "Assigning trust_score=0.0."
            )

        for tld in BLACKLISTED_TLDS:
            if host.endswith(tld):
                raise BlockedDomainError(
                    f"Domain '{host}' uses blacklisted TLD '{tld}'. "
                    "Assigning trust_score=0.0."
                )
    def _build_headers(self) -> dict[str, str]:
        """Build a realistic browser-like header dict with rotated UA and Referer."""
        return {
            "User-Agent": self._ua.random,
            "Referer": random.choice(_REFERERS),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "DNT": "1",
        }
    async def _fetch(self, url: str) -> str:
        self._validate_domain(url)

        if self._session is None or self._session.closed:
            raise RuntimeError(
                "BaseScraper._fetch called outside async context manager. "
                "Use: async with SomeScraper() as s: ..."
            )

        session = self._session  # local ref avoids closure issues
        @retry(
            retry=retry_if_exception_type(
                (aiohttp.ClientError, asyncio.TimeoutError)
            ),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _attempt() -> str:
            headers = self._build_headers()
            async with session.get(url, headers=headers, ssl=False) as resp:
                if resp.status in (429, 503):
                    # Treat rate-limit / service unavailable as retriable
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        status=resp.status,
                        message=f"HTTP {resp.status} — retrying with backoff",
                    )
                resp.raise_for_status()
                return await resp.text(errors="replace")
        try:
            return await _attempt()
        except Exception as exc:
            raise ScrapingError(
                f"All {self._max_retries} fetch attempts failed for '{url}': {exc}"
            ) from exc
    @abc.abstractmethod
    async def scrape(self, url: str) -> dict[str, Any]:
        """Scrape the given URL and return a structured result dict."""