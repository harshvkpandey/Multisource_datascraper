from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Any

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from scraper.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

YOUTUBE_DOMAIN_AUTHORITY: float = 0.90
LIKE_RATIO_CEILING: float = 0.10
VIEW_COUNT_CEILING: int = 10_000_000


def _extract_video_id(url: str) -> str | None:
    """Extract 11-char YouTube video ID from any supported URL format."""
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


class YouTubeScraper(BaseScraper):
    async def scrape(self, url: str) -> dict[str, Any]:
        #Scrape a YouTube video. Returns dict compatible with ScrapedItem schema.
        logger.info("[YouTubeScraper] Scraping: %s", url)
        self._validate_domain(url)

        video_id = _extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {url}")

        meta = self._extract_metadata(url)
        transcript = self._extract_transcript(video_id, fallback=meta.get("description", ""))
        author_credibility = self.compute_author_credibility(
            like_ratio=meta.get("like_ratio", 0.0),
            view_count=meta.get("view_count", 0),
        )

        return {
            "source_url": url,
            "source_type": "youtube",
            "title": meta.get("title"),
            "author": meta.get("channel") or meta.get("uploader"),
            "published_date": meta.get("upload_date"),
            "text": transcript,
            "domain": "youtube.com",
            "domain_authority": YOUTUBE_DOMAIN_AUTHORITY,
            "view_count": meta.get("view_count", 0),
            "like_count": meta.get("like_count", 0),
            "like_ratio": meta.get("like_ratio", 0.0),
            "author_credibility_proxy": author_credibility,
            "tags": meta.get("tags", []),
            "description": meta.get("description", ""),
            "citation_count": 0,
        }

    def _extract_metadata(self, url: str) -> dict[str, Any]:
        #Use yt-dlp to extract metadata without downloading the video
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": False,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False) or {}

                raw_date: str | None = info.get("upload_date")
                iso_date: str | None = None
                if raw_date:
                    try:
                        iso_date = datetime.strptime(raw_date, "%Y%m%d").isoformat()
                    except ValueError:
                        iso_date = raw_date

                view_count = int(info.get("view_count") or 0)
                like_count = int(info.get("like_count") or 0)
                like_ratio = like_count / view_count if view_count > 0 else 0.0

                return {
                    "title": info.get("title"),
                    "channel": info.get("channel") or info.get("uploader"),
                    "upload_date": iso_date,
                    "description": info.get("description", ""),
                    "view_count": view_count,
                    "like_count": like_count,
                    "like_ratio": round(like_ratio, 6),
                    "tags": list(info.get("tags") or []),
                }
        except Exception as exc:  # noqa: BLE001
            logger.error("[YouTubeScraper] yt-dlp error for %s: %s", url, exc)
            return {}

    def _extract_transcript(self, video_id: str, fallback: str) -> str:
        """Fetch transcript; fall back to description if unavailable."""
        try:
            segments = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["en", "en-US", "en-GB"]
            )
            transcript = " ".join(seg["text"] for seg in segments)
            logger.info("[YouTubeScraper] Transcript: %d chars for %s", len(transcript), video_id)
            return transcript
        except (TranscriptsDisabled, NoTranscriptFound):
            logger.warning("[YouTubeScraper] No transcript for %s; using description.", video_id)
            return fallback
        except Exception as exc:  # noqa: BLE001
            logger.error("[YouTubeScraper] Transcript error for %s: %s", video_id, exc)
            return fallback

    @staticmethod
    def compute_author_credibility(like_ratio: float, view_count: int) -> float:
        ratio_score = min(like_ratio / LIKE_RATIO_CEILING, 1.0)
        view_score = min(math.log1p(view_count) / math.log1p(VIEW_COUNT_CEILING), 1.0)
        return round(0.6 * ratio_score + 0.4 * view_score, 4)
