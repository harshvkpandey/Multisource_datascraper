"""
utils/chunking.py
=================
Recursive Character Text Chunker.

Splits text hierarchically using a priority-ordered separator list:
  \\n\\n → \\n → ". " → " " → ""  (character-level last resort)

Target chunk size : 512 characters
Overlap           : ~51 characters (10% of chunk_size)

The overlap ensures that downstream NLP models (e.g., embeddings,
zero-shot classifiers) do not lose context at chunk boundaries —
particularly important for sentence-straddling semantic units.

Design mirrors LangChain's RecursiveCharacterTextSplitter but is
implemented natively with no framework dependency.
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE: Final[int] = 512
DEFAULT_OVERLAP: Final[int] = 51          # ~10% of chunk size
DEFAULT_SEPARATORS: Final[list[str]] = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker:
    """
    Recursively splits text into overlapping chunks.

    Parameters
    ----------
    chunk_size : int
        Maximum length (in characters) of each chunk.
    overlap : int
        Number of characters shared between consecutive chunks.
    separators : list[str]
        Ordered list of split tokens. The chunker tries each separator
        in order, recursing into the next if chunks are still too large.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        separators: list[str] | None = None,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators if separators is not None else DEFAULT_SEPARATORS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def split(self, text: str) -> list[str]:
        """
        Split ``text`` into overlapping chunks of at most ``chunk_size``
        characters.

        Args:
            text: Raw input text (may contain newlines, multiple paragraphs).

        Returns:
            List[str]: Non-empty, whitespace-stripped chunks.
        """
        if not text or not text.strip():
            return []

        raw_chunks = self._split_recursive(text.strip(), self.separators)
        merged = self._merge_with_overlap(raw_chunks)
        result = [c.strip() for c in merged if c.strip()]
        logger.debug(
            "[Chunker] Produced %d chunks (size=%d, overlap=%d)",
            len(result),
            self.chunk_size,
            self.overlap,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: recursive splitting
    # ------------------------------------------------------------------
    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split ``text`` using the first separator that produces
        pieces small enough to fit within ``chunk_size``.

        If no separator reduces all pieces below ``chunk_size``, the final
        separator (empty string = character-level) is used as a hard fallback.
        """
        if not separators:
            # Character-level fallback: slice mechanically
            return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        remaining = separators[1:]

        # Split on the current separator
        splits = text.split(sep) if sep else list(text)
        splits = [s for s in splits if s.strip()]

        good_chunks: list[str] = []
        for piece in splits:
            if len(piece) <= self.chunk_size:
                good_chunks.append(piece)
            else:
                # Piece still too large — recurse with the next separator
                sub = self._split_recursive(piece, remaining)
                good_chunks.extend(sub)

        return good_chunks

    # ------------------------------------------------------------------
    # Internal: merge small pieces + add overlap
    # ------------------------------------------------------------------
    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """
        Greedily merge small pieces into chunks of ``chunk_size``, then
        apply a trailing ``overlap`` suffix from the previous chunk to the
        next one to preserve cross-boundary context.
        """
        chunks: list[str] = []
        current: list[str] = []
        current_len: int = 0

        for piece in pieces:
            piece_len = len(piece)

            if current_len + piece_len + 1 > self.chunk_size and current:
                # Flush current buffer as a chunk
                joined = " ".join(current)
                chunks.append(joined)

                # Retain overlap: take the tail of the flushed chunk
                tail = joined[-self.overlap:] if self.overlap > 0 else ""
                current = [tail] if tail else []
                current_len = len(tail)

            current.append(piece)
            current_len += piece_len + 1  # +1 for joining space

        if current:
            chunks.append(" ".join(current))

        return chunks
