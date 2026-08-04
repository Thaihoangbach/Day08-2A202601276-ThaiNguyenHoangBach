"""Task 6 - dependency-free BM25 lexical search.

This module uses the same chunks as Task 4.  Tokenization is Unicode-aware so
Vietnamese words and diacritics are retained instead of being stripped.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from threading import RLock

try:  # Support both ``python -m src.task6...`` and direct script execution.
    from .task4_chunking_indexing import chunk_documents, load_documents
except ImportError:  # pragma: no cover - used only by direct CLI execution
    from task4_chunking_indexing import chunk_documents, load_documents


CORPUS: list[dict] = []
_BM25_INDEX = None
_INDEX_SIGNATURE: tuple[str, ...] | None = None
_INDEX_LOCK = RLock()


def _tokenize(text: str) -> list[str]:
    """Normalize Unicode and return lowercase alphanumeric word tokens."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


class BM25Okapi:
    """Small BM25 Okapi implementation using k1=1.5 and b=0.75.

    The positive Robertson IDF variant avoids negative scores for terms that
    occur in more than half of a small corpus.
    """

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.corpus_size = len(tokenized_corpus)
        self.doc_len = [len(document) for document in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size else 0.0
        self.doc_freqs = [Counter(document) for document in tokenized_corpus]

        document_frequency: Counter[str] = Counter()
        for frequencies in self.doc_freqs:
            document_frequency.update(frequencies.keys())
        self.idf = {
            term: math.log(1.0 + (self.corpus_size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """Calculate a BM25 score for every document in corpus order."""
        if not self.corpus_size or not query_tokens:
            return [0.0] * self.corpus_size

        scores = [0.0] * self.corpus_size
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.doc_freqs):
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                length_ratio = self.doc_len[index] / self.avgdl if self.avgdl else 0.0
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                scores[index] += idf * (term_frequency * (self.k1 + 1.0)) / denominator
        return scores


def build_bm25_index(corpus: list[dict]):
    """Build and return a BM25 index from ``content`` in each corpus item."""
    if not isinstance(corpus, list):
        raise TypeError("corpus must be a list")

    tokenized_corpus: list[list[str]] = []
    for document in corpus:
        if not isinstance(document, dict) or not isinstance(document.get("content"), str):
            raise ValueError("each corpus item must contain a string 'content' field")
        if not isinstance(document.get("metadata", {}), dict):
            raise ValueError("corpus item 'metadata' must be a dictionary")
        tokenized_corpus.append(_tokenize(document["content"]))
    return BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)


def _ensure_default_corpus() -> None:
    if not CORPUS:
        CORPUS.extend(chunk_documents(load_documents()))


def _get_index():
    global _BM25_INDEX, _INDEX_SIGNATURE
    _ensure_default_corpus()
    signature = tuple(document.get("content", "") for document in CORPUS)
    with _INDEX_LOCK:
        if _BM25_INDEX is None or signature != _INDEX_SIGNATURE:
            _BM25_INDEX = build_bm25_index(CORPUS)
            _INDEX_SIGNATURE = signature
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return positive-scoring BM25 matches sorted in descending order."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    query = query.strip()
    if not query or top_k <= 0:
        return []

    bm25 = _get_index()
    if not CORPUS:
        return []
    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))

    results: list[dict] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0.0:
            break
        document = CORPUS[index]
        results.append(
            {
                "content": document["content"],
                "score": score,
                "metadata": dict(document.get("metadata") or {}),
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    matches = lexical_search("phương thức thanh toán shopee", top_k=5)
    for match in matches:
        print(f"[{match['score']:.3f}] {match['content'][:100]}...")
