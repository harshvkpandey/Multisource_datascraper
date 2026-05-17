"""
Formula: Trust Score = w1·A_cred + w2·C_score + w3·D_auth + w4·R + w5·M
Sub-score Definitions
A_cred  (Author Credibility)
    Derived from author presence, engagement metrics (YouTube), or
    publication record (PubMed). Missing author → 0.0 + penalty log.
    High text subjectivity (TextBlob > 0.6) applies a 0.5× multiplier
    per the E-E-A-T NLP abuse-prevention framework.
C_score (Citation Score)  — PageRank bibliometric proxy
    C_score = ln(1 + C) / ln(1 + C_max)
    Log-normalisation prevents citation outliers from dominating.
    Only meaningful for PubMed; set to 0.0 for blog/YouTube.
D_auth  (Domain Authority)  — TrustRank-inspired
    Binary/ordinal check against a curated seed whitelist:
    1.0 = verified trusted seed domain
    0.8 = subdomain of trusted seed
    0.9 = .edu / .gov TLD
    0.5 = unknown / unverified domain
R       (Recency)  — Exponential decay
    R = e^(-lambda * t)   where t = days since publication
    lambda (RECENCY_LAMBDA) defaults to 0.005:
      t=0   → R ≈ 1.000  (today)
      t=365 → R ≈ 0.160  (one year old)
      t=730 → R ≈ 0.026  (two years old)
    Missing date → R = 0.30 (fixed penalty).
M       (Medical/YMYL Modifier)  — Google E-E-A-T
    Activated when topic_tags contain health/medical terms.
    Disclaimer present  → M = 1.0 (no penalty)
    Disclaimer absent   → M = 0.25 (severe penalty; YMYL protection)
    Non-medical content → M = 1.0 (neutral multiplier)
Dynamic Weights (sum = 1.0 per source type)
              A_cred  C_score  D_auth  R      M
blog           0.25    0.05    0.35   0.20   0.15
youtube        0.30    0.05    0.25   0.25   0.15
pubmed         0.20    0.35    0.15   0.15   0.15 (wait, 0.20+0.35+0.15+0.15+0.15=1.0 ✓)

Abuse Prevention Multipliers (post-score, multiplicative)
- Blacklisted domain           → trust_score = 0.0  (hard zero)
- High subjectivity (> 0.6)    → A_cred × 0.5
- Missing published_date       → R = 0.30 (fixed)
- Missing author               → A_cred = 0.0
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Final, Literal

logger = logging.getLogger(__name__)
RECENCY_LAMBDA: float = float(os.getenv("RECENCY_LAMBDA", "0.005"))
CITATION_COUNT_MAX: int = int(os.getenv("CITATION_COUNT_MAX", "50000"))
# Subjectivity multiplier threshold (mirrors tagging.py SUBJECTIVITY_THRESHOLD)
SUBJECTIVITY_MULTIPLIER_THRESHOLD: float = 0.60
SUBJECTIVITY_MULTIPLIER: float = 0.50
# Medical / YMYL topic keywords (E-E-A-T YMYL detection)
_MEDICAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "healthcare", "medicine", "medical", "health", "clinical", "disease",
        "drug", "treatment", "symptom", "diagnosis", "therapy", "pharmaceutical",
        "pubmed", "genomic", "neuroscience", "cancer", "vaccine", "surgery",
        "mental health", "nutrition", "biology",
    }
)
# Medical disclaimer search strings
_DISCLAIMER_PHRASES: tuple[str, ...] = (
    "not medical advice",
    "consult a physician",
    "consult your doctor",
    "for informational purposes only",
    "medical disclaimer",
    "speak to a healthcare",
    "this is not intended",
    "always seek professional",
)

SourceType = Literal["blog", "youtube", "pubmed"]
_WEIGHTS: dict[SourceType, dict[str, float]] = {
    "blog": {
        "a_cred": 0.25,
        "c_score": 0.05,
        "d_auth": 0.35,
        "r": 0.20,
        "m": 0.15,
    },
    "youtube": {
        "a_cred": 0.30,
        "c_score": 0.05,
        "d_auth": 0.25,
        "r": 0.25,
        "m": 0.15,
    },
    "pubmed": {
        "a_cred": 0.20,
        "c_score": 0.35,
        "d_auth": 0.15,
        "r": 0.15,
        "m": 0.15,
    },
}
def compute_recency(published_date: str | None) -> float:
    """
    Returns:
        float in [0.0, 1.0].
            0.30 if date is missing (penalty).
            ~1.0 if published today.
            ~0.16 if published 365 days ago (with lambda=0.005).
    """
    if not published_date:
        logger.warning("[TrustScore] Missing published_date — applying R=0.30 penalty.")
        return 0.30
    try:
        # Handle ISO strings with or without timezone
        pub = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        t_days = max((now - pub).total_seconds() / 86_400, 0.0)
        score = math.exp(-RECENCY_LAMBDA * t_days)
        return round(min(max(score, 0.0), 1.0), 6)
    except (ValueError, OverflowError) as exc:
        logger.warning("[TrustScore] Cannot parse date '%s': %s. R=0.30.", published_date, exc)
        return 0.30


def compute_citation_score(citation_count: int, c_max: int = CITATION_COUNT_MAX) -> float:
    """
    Returns:
        float in [0.0, 1.0].
    """
    if citation_count <= 0:
        return 0.0
    c_max = max(c_max, 1)
    score = math.log1p(citation_count) / math.log1p(c_max)
    return round(min(max(score, 0.0), 1.0), 6)
def compute_author_credibility(
    raw_credibility: float,
    author_present: bool,
    subjectivity_score: float,
) -> float:
    """
    Returns:
        float in [0.0, 1.0].
    """
    if not author_present:
        logger.warning("[TrustScore] No author detected — A_cred = 0.0.")
        return 0.0
    score = raw_credibility
    if subjectivity_score > SUBJECTIVITY_MULTIPLIER_THRESHOLD:
        logger.warning(
            "[TrustScore] High subjectivity (%.3f > %.2f) — "
            "applying A_cred × %.1f multiplier (E-E-A-T SEO flag).",
            subjectivity_score,
            SUBJECTIVITY_MULTIPLIER_THRESHOLD,
            SUBJECTIVITY_MULTIPLIER,
        )
        score *= SUBJECTIVITY_MULTIPLIER
    return round(min(max(score, 0.0), 1.0), 6)
def compute_medical_modifier(
    topic_tags: list[str],
    content_text: str,
    source_type: SourceType,
) -> float:
    """
    Returns:
        float: M ∈ {0.25, 1.0}.
    """
    # PubMed is inherently medical; always check for disclaimer
    tags_lower = {t.lower() for t in topic_tags}
    is_medical = (
        source_type == "pubmed"
        or bool(tags_lower & _MEDICAL_KEYWORDS)
        or any(kw in t for t in tags_lower for kw in _MEDICAL_KEYWORDS)
    )
    if not is_medical:
        return 1.0
    text_lower = content_text.lower()
    has_disclaimer = any(phrase in text_lower for phrase in _DISCLAIMER_PHRASES)

    if has_disclaimer:
        logger.debug("[TrustScore] YMYL content — disclaimer found. M = 1.0.")
        return 1.0

    logger.warning(
        "[TrustScore] YMYL/medical content detected but NO disclaimer found — "
        "applying severe M = 0.25 penalty (E-E-A-T protection)."
    )
    return 0.25

def compute_trust_score(
    source_type: SourceType,
    *,
    author: str | None,
    published_date: str | None,
    domain_authority: float,
    citation_count: int,
    raw_author_credibility: float,
    subjectivity_score: float,
    topic_tags: list[str],
    content_text: str,
    is_blacklisted: bool = False,
) -> float:
    """
    Returns:
        float: Trust score in [0.0, 1.0], rounded to 4 decimal places.
    """
    # Hard zero for blacklisted domains (TrustRank enforcement)
    if is_blacklisted:
        logger.warning("[TrustScore] Blacklisted domain — trust_score = 0.0.")
        return 0.0

    weights = _WEIGHTS.get(source_type, _WEIGHTS["blog"])

    # Sub-score computation
    a_cred = compute_author_credibility(
        raw_credibility=raw_author_credibility,
        author_present=bool(author and author.strip()),
        subjectivity_score=subjectivity_score,
    )
    c_score = compute_citation_score(citation_count)
    d_auth = round(min(max(float(domain_authority), 0.0), 1.0), 6)
    r_score = compute_recency(published_date)
    m_modifier = compute_medical_modifier(topic_tags, content_text, source_type)

    # Weighted sum
    score = (
        weights["a_cred"] * a_cred
        + weights["c_score"] * c_score
        + weights["d_auth"] * d_auth
        + weights["r"] * r_score
        + weights["m"] * m_modifier
    )
    # Final clamp to [0, 1] (defensive; formula should never exceed bounds)
    final = round(min(max(score, 0.0), 1.0), 4)

    logger.info(
        "[TrustScore] %s | A_cred=%.4f C_score=%.4f D_auth=%.4f R=%.4f M=%.4f → %.4f",
        source_type, a_cred, c_score, d_auth, r_score, m_modifier, final,
    )
    return final

def get_weights(source_type: SourceType) -> dict[str, float]:
    """Return the weight table for a given source type (for testing/inspection)."""
    return dict(_WEIGHTS.get(source_type, _WEIGHTS["blog"]))
