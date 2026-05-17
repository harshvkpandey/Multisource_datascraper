"""
models.py
=========
Pydantic V2 data validation schema for the pipeline output.

Every scraped item is validated against ``ScrapedItem`` before being
written to ``output/scraped_data.json``. This ensures:
  - Type correctness (str, float, list, etc.)
  - URL format validation (HttpUrl)
  - trust_score clamped to [0.0, 1.0] via field_validator
  - source_type constrained to a Literal union
  - ISO language code populated by langdetect
"""

from __future__ import annotations

import logging
from typing import Annotated, List, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

logger = logging.getLogger(__name__)

SourceTypeLiteral = Literal["blog", "youtube", "pubmed"]


class ScrapedItem(BaseModel):
    """
    Validated output schema for a single scraped source.

    Fields
    ------
    source_url      HTTP/HTTPS URL — Pydantic HttpUrl validation.
    source_type     Constrained to 'blog', 'youtube', or 'pubmed'.
    author          Author name, channel name, or 'Unknown' if absent.
    published_date  ISO 8601 string (e.g. '2023-04-15T00:00:00') or None.
    language        ISO 639-1 code auto-detected by langdetect (e.g. 'en').
    region          Geographic region code or None if not determinable.
    topic_tags      Zero-shot NLI-generated topic labels (min 0, no hard cap).
    trust_score     Weighted algorithm output, always in [0.0, 1.0].
    content_chunks  Recursive character-split text segments.
    """

    source_url: str = Field(..., description="Validated HTTP/HTTPS source URL")
    source_type: SourceTypeLiteral = Field(
        ..., description="Content origin type: blog | youtube | pubmed"
    )
    author: str = Field(
        default="Unknown",
        description="Author, channel name, or organisation",
    )
    published_date: str | None = Field(
        default=None,
        description="ISO 8601 publication date string or null",
    )
    language: str = Field(
        default="en",
        description="ISO 639-1 language code auto-detected from content",
    )
    region: str | None = Field(
        default=None,
        description="Geographic region code (e.g. 'US') or null",
    )
    topic_tags: List[str] = Field(
        default_factory=list,
        description="Zero-shot NLI-classified topic labels",
    )
    trust_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Weighted trust score in [0.0, 1.0]",
    )
    content_chunks: List[str] = Field(
        default_factory=list,
        description="Recursively split content segments (~512 chars, 10% overlap)",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("source_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure source_url is a valid HTTP/HTTPS URL."""
        from urllib.parse import urlparse
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"source_url must use http or https scheme, got: '{v}'")
        if not parsed.netloc:
            raise ValueError(f"source_url has no host: '{v}'")
        return v

    @field_validator("trust_score")
    @classmethod
    def clamp_trust_score(cls, v: float) -> float:
        """Hard-clamp trust_score to [0.0, 1.0] as a safety net."""
        clamped = round(min(max(float(v), 0.0), 1.0), 4)
        if clamped != v:
            logger.warning(
                "[Schema] trust_score %.6f out of bounds — clamped to %.4f", v, clamped
            )
        return clamped

    @field_validator("language")
    @classmethod
    def normalise_language(cls, v: str) -> str:
        """Normalise language code to lowercase."""
        return v.lower().strip() if v else "en"

    @field_validator("author")
    @classmethod
    def default_unknown_author(cls, v: str) -> str:
        """Replace empty/whitespace-only author with 'Unknown'."""
        return v.strip() if v and v.strip() else "Unknown"

    @model_validator(mode="after")
    def warn_empty_chunks(self) -> "ScrapedItem":
        """Warn if content_chunks is empty (indicates scraping failure)."""
        if not self.content_chunks:
            logger.warning(
                "[Schema] content_chunks is empty for %s — "
                "text extraction may have failed.",
                self.source_url,
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_url": "https://karpathy.github.io/2015/05/21/rnn-effectiveness/",
                "source_type": "blog",
                "author": "Andrej Karpathy",
                "published_date": "2015-05-21T00:00:00",
                "language": "en",
                "region": None,
                "topic_tags": ["Artificial Intelligence", "Machine Learning"],
                "trust_score": 0.8742,
                "content_chunks": ["The Unreasonable Effectiveness of Recurrent..."],
            }
        }
    }
