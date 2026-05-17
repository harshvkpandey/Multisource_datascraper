

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import pytest

# Scoring engine
from scoring.trust_score import (
    compute_citation_score,
    compute_recency,
    compute_author_credibility,
    compute_medical_modifier,
    compute_trust_score,
    get_weights,
    CITATION_COUNT_MAX,
    RECENCY_LAMBDA,
)

# Chunker
from utils.chunking import RecursiveChunker

# Pydantic schema
from models import ScrapedItem
class TestWeights:
    """All weight tables must sum to exactly 1.0."""

    @pytest.mark.parametrize("source_type", ["blog", "youtube", "pubmed"])
    def test_weights_sum_to_one(self, source_type: str) -> None:
        weights = get_weights(source_type)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, (
            f"Weights for '{source_type}' sum to {total}, expected 1.0"
        )

    @pytest.mark.parametrize("source_type", ["blog", "youtube", "pubmed"])
    def test_all_weights_non_negative(self, source_type: str) -> None:
        for k, v in get_weights(source_type).items():
            assert v >= 0.0, f"Weight '{k}' for '{source_type}' is negative: {v}"
class TestCitationScore:
    def test_zero_citations_returns_zero(self) -> None:
        assert compute_citation_score(0) == 0.0

    def test_c_max_returns_one(self) -> None:
        score = compute_citation_score(CITATION_COUNT_MAX)
        assert abs(score - 1.0) < 1e-6, f"Expected ~1.0, got {score}"

    def test_monotonically_increasing(self) -> None:
        """Higher citation counts must yield higher C_scores."""
        scores = [compute_citation_score(c) for c in [0, 10, 100, 1000, 10000]]
        assert scores == sorted(scores), f"C_score not monotone: {scores}"

    def test_bounded_between_zero_and_one(self) -> None:
        for c in [0, 1, 50, 500, 5000, 50000, 100000, 1_000_000]:
            s = compute_citation_score(c)
            assert 0.0 <= s <= 1.0, f"C_score={s} out of bounds for C={c}"

    def test_log_normalisation_formula(self) -> None:
        c = 1000
        expected = math.log1p(c) / math.log1p(CITATION_COUNT_MAX)
        assert abs(compute_citation_score(c) - expected) < 1e-6

    def test_negative_count_returns_zero(self) -> None:
        assert compute_citation_score(-5) == 0.0
class TestRecency:
    def test_today_returns_near_one(self) -> None:
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        r = compute_recency(now_iso)
        assert r > 0.99, f"Recency for today should be ~1.0, got {r}"

    def test_missing_date_returns_penalty(self) -> None:
        assert compute_recency(None) == 0.30

    def test_one_year_old_within_range(self) -> None:
        one_year_ago = (datetime.now(tz=timezone.utc) - timedelta(days=365)).isoformat()
        r = compute_recency(one_year_ago)
        expected = math.exp(-RECENCY_LAMBDA * 365)
        assert abs(r - expected) < 0.001, f"Expected ~{expected:.4f}, got {r}"

    def test_always_bounded(self) -> None:
        dates = [
            datetime.now(tz=timezone.utc).isoformat(),
            (datetime.now(tz=timezone.utc) - timedelta(days=100)).isoformat(),
            (datetime.now(tz=timezone.utc) - timedelta(days=3650)).isoformat(),
            None,
            "not-a-date",
        ]
        for d in dates:
            r = compute_recency(d)
            assert 0.0 <= r <= 1.0, f"R={r} out of bounds for date={d!r}"

    def test_older_date_lower_recency(self) -> None:
        recent = (datetime.now(tz=timezone.utc) - timedelta(days=30)).isoformat()
        older = (datetime.now(tz=timezone.utc) - timedelta(days=365)).isoformat()
        assert compute_recency(recent) > compute_recency(older)

    def test_future_date_clamps_to_one(self) -> None:
        future = (datetime.now(tz=timezone.utc) + timedelta(days=30)).isoformat()
        r = compute_recency(future)
        assert r <= 1.0

class TestAuthorCredibility:
    def test_missing_author_returns_zero(self) -> None:
        score = compute_author_credibility(
            raw_credibility=0.9, author_present=False, subjectivity_score=0.1
        )
        assert score == 0.0

    def test_high_subjectivity_halves_score(self) -> None:
        normal = compute_author_credibility(
            raw_credibility=0.8, author_present=True, subjectivity_score=0.3
        )
        penalised = compute_author_credibility(
            raw_credibility=0.8, author_present=True, subjectivity_score=0.7
        )
        assert penalised == pytest.approx(normal * 0.5, abs=1e-4)

    def test_normal_case_passes_through(self) -> None:
        score = compute_author_credibility(
            raw_credibility=0.75, author_present=True, subjectivity_score=0.2
        )
        assert score == pytest.approx(0.75, abs=1e-4)

    def test_always_bounded(self) -> None:
        for raw, present, subj in [
            (1.0, True, 0.0),
            (1.0, True, 0.9),
            (0.0, False, 0.5),
            (0.5, True, 0.61),
        ]:
            s = compute_author_credibility(raw, present, subj)
            assert 0.0 <= s <= 1.0

class TestMedicalModifier:
    def test_non_medical_returns_one(self) -> None:
        m = compute_medical_modifier(["Technology", "Software Engineering"], "some text", "blog")
        assert m == 1.0

    def test_medical_tag_with_disclaimer_returns_one(self) -> None:
        m = compute_medical_modifier(
            ["Healthcare"],
            "This article is for informational purposes only.",
            "blog",
        )
        assert m == 1.0

    def test_medical_tag_no_disclaimer_returns_penalty(self) -> None:
        m = compute_medical_modifier(
            ["Healthcare"],
            "Here is some health advice.",
            "blog",
        )
        assert m == 0.25

    def test_pubmed_always_medical(self) -> None:
        """PubMed source type should always trigger YMYL check."""
        m_no_disclaimer = compute_medical_modifier([], "Abstract content.", "pubmed")
        assert m_no_disclaimer == 0.25

    def test_pubmed_with_disclaimer(self) -> None:
        m = compute_medical_modifier(
            [],
            "Consult a physician before acting on this information.",
            "pubmed",
        )
        assert m == 1.0

class TestTrustScoreIntegration:
    def _base_kwargs(self, source_type: str = "blog") -> dict:
        return dict(
            source_type=source_type,
            author="Jane Doe",
            published_date=datetime.now(tz=timezone.utc).isoformat(),
            domain_authority=0.8,
            citation_count=100,
            raw_author_credibility=0.75,
            subjectivity_score=0.2,
            topic_tags=["Technology"],
            content_text="A well-written, factual article about machine learning.",
            is_blacklisted=False,
        )

    def test_score_in_zero_one(self) -> None:
        for st in ["blog", "youtube", "pubmed"]:
            kwargs = self._base_kwargs(st)
            score = compute_trust_score(**kwargs)
            assert 0.0 <= score <= 1.0, f"Score={score} out of bounds for {st}"

    def test_blacklisted_domain_returns_zero(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["is_blacklisted"] = True
        assert compute_trust_score(**kwargs) == 0.0

    def test_pubmed_high_citations_raises_score(self) -> None:
        low = compute_trust_score(**{**self._base_kwargs("pubmed"), "citation_count": 0})
        high = compute_trust_score(**{**self._base_kwargs("pubmed"), "citation_count": 5000})
        assert high > low, "Higher citations should raise PubMed trust score"

    def test_older_content_lower_score(self) -> None:
        recent_date = datetime.now(tz=timezone.utc).isoformat()
        old_date = (datetime.now(tz=timezone.utc) - timedelta(days=1000)).isoformat()
        recent = compute_trust_score(**{**self._base_kwargs(), "published_date": recent_date})
        old = compute_trust_score(**{**self._base_kwargs(), "published_date": old_date})
        assert recent > old

    def test_missing_author_reduces_score(self) -> None:
        with_author = compute_trust_score(**self._base_kwargs())
        kwargs = {**self._base_kwargs(), "author": None}
        without_author = compute_trust_score(**kwargs)
        assert with_author > without_author

    def test_ymyl_no_disclaimer_reduces_score(self) -> None:
        safe = compute_trust_score(
            **{**self._base_kwargs(), "topic_tags": [], "content_text": "neutral text"}
        )
        risky = compute_trust_score(
            **{
                **self._base_kwargs(),
                "topic_tags": ["Healthcare"],
                "content_text": "Health advice without disclaimer.",
            }
        )
        assert safe > risky

class TestRecursiveChunker:
    def test_empty_text_returns_empty(self) -> None:
        c = RecursiveChunker()
        assert c.split("") == []
        assert c.split("   ") == []

    def test_short_text_returns_single_chunk(self) -> None:
        c = RecursiveChunker(chunk_size=512)
        text = "Short text."
        chunks = c.split(text)
        assert len(chunks) == 1
        assert chunks[0].strip() == text.strip()

    def test_chunk_sizes_within_bounds(self) -> None:
        c = RecursiveChunker(chunk_size=512, overlap=51)
        text = " ".join(["word"] * 2000)
        chunks = c.split(text)
        # Allow small overrun from overlap merging
        for chunk in chunks:
            assert len(chunk) <= 600, f"Chunk too large: {len(chunk)}"

    def test_overlap_shared_content(self) -> None:
        """Consecutive chunks must share some content (non-zero overlap)."""
        c = RecursiveChunker(chunk_size=100, overlap=20)
        text = "A" * 90 + " " + "B" * 90 + " " + "C" * 90
        chunks = c.split(text)
        if len(chunks) >= 2:
            # Tail of chunk N should appear somewhere in chunk N+1
            tail = chunks[0][-20:]
            assert tail in chunks[1], "Expected overlap content in consecutive chunks"

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=100, overlap=100)

class TestScrapedItemSchema:
    def _valid_item(self) -> dict:
        return {
            "source_url": "https://example.com/article",
            "source_type": "blog",
            "author": "Test Author",
            "published_date": "2024-01-01T00:00:00",
            "language": "en",
            "region": None,
            "topic_tags": ["Technology"],
            "trust_score": 0.75,
            "content_chunks": ["chunk one", "chunk two"],
        }

    def test_valid_item_passes(self) -> None:
        item = ScrapedItem(**self._valid_item())
        assert item.trust_score == 0.75

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(Exception):
            ScrapedItem(**{**self._valid_item(), "source_url": "not-a-url"})

    def test_trust_score_clamped_above_one(self) -> None:
        item = ScrapedItem(**{**self._valid_item(), "trust_score": 1.5})
        assert item.trust_score == 1.0

    def test_trust_score_clamped_below_zero(self) -> None:
        item = ScrapedItem(**{**self._valid_item(), "trust_score": -0.5})
        assert item.trust_score == 0.0

    def test_invalid_source_type_raises(self) -> None:
        with pytest.raises(Exception):
            ScrapedItem(**{**self._valid_item(), "source_type": "twitter"})

    def test_empty_author_defaults_to_unknown(self) -> None:
        item = ScrapedItem(**{**self._valid_item(), "author": "   "})
        assert item.author == "Unknown"

    def test_language_normalised_to_lowercase(self) -> None:
        item = ScrapedItem(**{**self._valid_item(), "language": "EN"})
        assert item.language == "en"
