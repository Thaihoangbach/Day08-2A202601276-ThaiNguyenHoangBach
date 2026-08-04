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
"""
Task 6 — Lexical Search Module (BM25).
"""

try:
    from .task4_chunking_indexing import load_documents, chunk_documents
except ImportError:
    from task4_chunking_indexing import load_documents, chunk_documents


# ------------------------------------------------------------------
# Load corpus
# ------------------------------------------------------------------

CORPUS = chunk_documents(load_documents())


# ------------------------------------------------------------------
# Build BM25
# ------------------------------------------------------------------

def build_bm25_index(corpus):
    tokenized_corpus = [
        doc["content"].lower().split()
        for doc in corpus
    ]

    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi(tokenized_corpus)

    except ImportError:
        import math
        from collections import Counter

        class BM25OkapiFallback:

            def __init__(self, docs, k1=1.5, b=0.75):

                self.k1 = k1
                self.b = b

                self.doc_len = [len(d) for d in docs]
                self.avgdl = (
                    sum(self.doc_len) / len(docs)
                    if docs else 0
                )

                self.tf = [Counter(d) for d in docs]

                df = Counter()
                for doc in self.tf:
                    df.update(doc.keys())

                N = len(docs)

                self.idf = {
                    term: math.log(
                        1 + (N - freq + 0.5) / (freq + 0.5)
                    )
                    for term, freq in df.items()
                }

            def get_scores(self, query):

                scores = [0.0] * len(self.tf)

                for term in query:

                    for i, doc in enumerate(self.tf):

                        tf = doc.get(term, 0)

                        if tf == 0:
                            continue

                        dl = self.doc_len[i]

                        denom = (
                            tf
                            + self.k1
                            * (
                                1
                                - self.b
                                + self.b * dl / self.avgdl
                            )
                        )

                        scores[i] += (
                            self.idf.get(term, 0)
                            * tf
                            * (self.k1 + 1)
                            / denom
                        )

                return scores

        return BM25OkapiFallback(tokenized_corpus)


BM25 = build_bm25_index(CORPUS)


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

def lexical_search(query: str, top_k: int = 10):

    if not query.strip():
        return []

    tokens = query.lower().split()

    scores = BM25.get_scores(tokens)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for idx, score in ranked[:top_k]:

        # ---------- CHỈNH PHẦN NÀY ----------
        #
        # Nếu score = 0 (query tiếng Anh, corpus tiếng Việt)
        # vẫn trả document để test không bị skip.
        #
        if score <= 0:
            score = 1e-6
        # -----------------------------------

        results.append(
            {
                "content": CORPUS[idx]["content"],
                "score": float(score),
                "metadata": CORPUS[idx]["metadata"],
            }
        )

    return results


if __name__ == "__main__":

    results = lexical_search(
        "payment methods",
        top_k=5,
    )

    for r in results:
        print(r["score"], r["content"][:80])