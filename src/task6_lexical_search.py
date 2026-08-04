"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

# Load corpus từ đúng pipeline load/chunk của Task 4 để dense và lexical search
# cùng tìm kiếm trên một tập chunks và dùng chung metadata.
try:
    from .task4_chunking_indexing import load_documents, chunk_documents
except ImportError:
    from task4_chunking_indexing import load_documents, chunk_documents

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
CORPUS.extend(chunk_documents(load_documents()))


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # Dùng rank-bm25 khi đã cài; fallback giữ nguyên công thức BM25 Okapi để
    # module vẫn chạy được trong môi trường kiểm thử tối giản.
    #
    # from rank_bm25 import BM25Okapi
    #
    # # Tokenize - có thể đơn giản split(), hoặc dùng underthesea cho tiếng Việt
    # tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    # bm25 = BM25Okapi(tokenized_corpus)
    # return bm25
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(tokenized_corpus)
    except ImportError:
        import math
        from collections import Counter

        class BM25OkapiFallback:
            def __init__(self, documents, k1=1.5, b=0.75):
                self.k1 = k1
                self.b = b
                self.lengths = [len(doc) for doc in documents]
                self.avgdl = sum(self.lengths) / len(documents) if documents else 0.0
                self.frequencies = [Counter(doc) for doc in documents]
                document_frequencies = Counter()
                for frequencies in self.frequencies:
                    document_frequencies.update(frequencies.keys())
                count = len(documents)
                self.idf = {
                    term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
                    for term, frequency in document_frequencies.items()
                }

            def get_scores(self, query_tokens):
                scores = [0.0] * len(self.frequencies)
                for term in query_tokens:
                    for i, frequencies in enumerate(self.frequencies):
                        frequency = frequencies.get(term, 0)
                        if not frequency:
                            continue
                        length_ratio = self.lengths[i] / self.avgdl if self.avgdl else 0.0
                        denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                        scores[i] += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
                return scores

        return BM25OkapiFallback(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    # Chấm BM25, lấy tối đa top_k kết quả dương và giữ thứ tự score giảm dần.
    #
    # tokenized_query = query.lower().split()
    # scores = bm25.get_scores(tokenized_query)
    #
    # # Get top_k indices
    # import numpy as np
    # top_indices = np.argsort(scores)[::-1][:top_k]
    #
    # results = []
    # for idx in top_indices:
    #     if scores[idx] > 0:
    #         results.append({
    #             "content": CORPUS[idx]["content"],
    #             "score": float(scores[idx]),
    #             "metadata": CORPUS[idx]["metadata"]
    #         })
    # return results
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if not query.strip() or top_k <= 0 or not CORPUS:
        return []

    bm25 = build_bm25_index(CORPUS)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
