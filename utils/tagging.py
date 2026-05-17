from __future__ import annotations

import logging
import os
from typing import Final

logger = logging.getLogger(__name__)

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
    def _get_pipeline(self):
        #Load HuggingFace zero-shot classification pipeline on first call.
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
    def get_tags(self, text: str) -> list[str]:
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
    @staticmethod
    def get_subjectivity(text: str) -> float:
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
        #Return True if the subjectivity score exceeds the abuse threshold
        return subjectivity_score > SUBJECTIVITY_THRESHOLD
