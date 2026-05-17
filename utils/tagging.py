"""
utils/tagging.py
================
Zero-Shot Topic Tagger and NLP Subjectivity Analyser.

Zero-Shot Classification
------------------------
Uses a Hugging Face NLI model to assign topic tags without training data.
Default model : cross-encoder/nli-MiniLM2-L6-H4  (~80 MB, fast)
Upgrade model : facebook/bart-large-mnli           (~1.6 GB, higher accuracy)

Set TAGGING_MODEL env var to switch. The model is downloaded once and
cached by HuggingFace in ~/.cache/huggingface/hub/.

NLP Subjectivity Analysis (Abuse Prevention)
--------------------------------------------
Uses TextBlob's sentiment lexicon to compute a subjectivity score ∈ [0, 1].
  0.0 = fully objective
  1.0 = fully subjective / emotional

If subjectivity > 0.6, the scoring engine applies a 0.5× multiplier to
A_cred, flagging content as potential SEO spam or opinion-heavy material.
This implements the E-E-A-T principle: Experience and Expertise are
undermined by highly emotional, non-factual writing style.
"""

from __future__ import annotations

import logging
import os
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_MODEL_NAME: str = os.getenv("TAGGING_MODEL", "cross-encoder/nli-MiniLM2-L6-H4")

# Candidate topic labels for zero-shot classification
CANDIDATE_LABELS: Final[list[str]] = [
    "Artificial Intelligence",
    "Machine Learning",
    "Healthcare",
    "Data Science",
    "Climate Science",
    "Economics",
    "Technology",
    "Politics",
    "Education",
    "Research",
    "Neuroscience",
    "Mathematics",
    "Software Engineering",
    "Ethics",
]

# Only return labels whose NLI entailment score exceeds this threshold
CONFIDENCE_THRESHOLD: Final[float] = 0.20

# Maximum number of topic tags to return per item
MAX_TAGS: Final[int] = 5

# Subjectivity threshold for abuse-prevention multiplier
SUBJECTIVITY_THRESHOLD: Final[float] = 0.60


class ZeroShotTagger:
    """
    Assigns topic tags to text using zero-shot NLI classification.

    The pipeline is lazily initialised on first use to avoid loading
    ~80–1600 MB of model weights at import time.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        candidate_labels: list[str] | None = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_tags: int = MAX_TAGS,
    ) -> None:
        self._model_name = model_name
        self._labels = candidate_labels or CANDIDATE_LABELS
        self._threshold = confidence_threshold
        self._max_tags = max_tags
        self._pipeline = None  # lazy load

    # ------------------------------------------------------------------
    # Lazy pipeline initialisation
    # ------------------------------------------------------------------
    def _get_pipeline(self):
        """Load HuggingFace zero-shot classification pipeline on first call."""
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline  # lazy import

            logger.info("[Tagger] Loading model '%s' …", self._model_name)
            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=-1,       # CPU; set to 0 for CUDA GPU
            )
            logger.info("[Tagger] Model loaded.")
        return self._pipeline

    # ------------------------------------------------------------------
    # Public: topic tagging
    # ------------------------------------------------------------------
    def get_tags(self, text: str) -> list[str]:
        """
        Run zero-shot classification on ``text`` and return topic labels.

        For efficiency, classifies only the first 512 characters of the
        joined text (fast approximation; sufficient for topic detection).

        Args:
            text: Article or transcript body text.

        Returns:
            List[str]: Up to ``max_tags`` labels with score > threshold.
        """
        if not text or not text.strip():
            logger.warning("[Tagger] Empty text — returning empty tag list.")
            return []

        snippet = text.strip()[:512]

        try:
            pipe = self._get_pipeline()
            result = pipe(snippet, self._labels, multi_label=True)

            tags: list[str] = [
                label
                for label, score in zip(result["labels"], result["scores"])
                if score >= self._threshold
            ][: self._max_tags]

            logger.debug("[Tagger] Tags: %s", tags)
            return tags

        except Exception as exc:  # noqa: BLE001
            logger.error("[Tagger] Zero-shot classification failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Public: subjectivity analysis (E-E-A-T abuse prevention)
    # ------------------------------------------------------------------
    @staticmethod
    def get_subjectivity(text: str) -> float:
        """
        Compute text subjectivity score using TextBlob's sentiment lexicon.

        TextBlob.sentiment.subjectivity returns a float in [0.0, 1.0]:
          - 0.0  : completely objective (factual, neutral)
          - 1.0  : completely subjective (emotional, opinionated)

        Scores above SUBJECTIVITY_THRESHOLD (0.6) trigger a 0.5×
        A_cred multiplier in the trust scoring engine, flagging content
        that reads more like opinion or SEO-bait than authoritative prose.

        Args:
            text: Article body text.

        Returns:
            float: Subjectivity score in [0.0, 1.0].
        """
        if not text or not text.strip():
            return 0.0
        try:
            from textblob import TextBlob  # lazy import

            snippet = text.strip()[:2000]  # analyse first 2000 chars
            score = float(TextBlob(snippet).sentiment.subjectivity)
            logger.debug("[Tagger] Subjectivity score: %.3f", score)
            return round(score, 4)

        except Exception as exc:  # noqa: BLE001
            logger.error("[Tagger] Subjectivity analysis failed: %s", exc)
            return 0.0

    @staticmethod
    def is_high_subjectivity(subjectivity_score: float) -> bool:
        """Return True if the subjectivity score exceeds the abuse threshold."""
        return subjectivity_score > SUBJECTIVITY_THRESHOLD
