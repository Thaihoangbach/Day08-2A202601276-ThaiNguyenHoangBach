"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Hybrid Retrieval Pipeline.

    Pipeline
    --------
    1. Semantic Search
    2. Lexical Search (BM25)
    3. RRF Fusion
    4. Reranking
    5. PageIndex fallback nếu semantic score quá thấp
    """

    # ==========================================================
    # Step 1. Retrieval
    # ==========================================================

    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # ==========================================================
    # Step 2. RRF Fusion
    # ==========================================================

    merged = rerank_rrf(
        [dense_results, sparse_results],
        top_k=top_k * 2,
    )

    for item in merged:
        item["source"] = "hybrid"

    # ==========================================================
    # Step 3. Rerank
    # ==========================================================

    if use_reranking and merged:
        final_results = rerank(
            query,
            merged,
            top_k=top_k,
            method=RERANK_METHOD,
        )
    else:
        final_results = merged[:top_k]

    for item in final_results:
        item.setdefault("source", "hybrid")

    # ==========================================================
    # Step 4. Fallback
    # ==========================================================

    best_score = (
        dense_results[0]["score"]
        if dense_results
        else 0.0
    )

    if best_score < score_threshold:

        print(
            f"  ⚠ Semantic best score ({best_score:.3f}) "
            f"< threshold ({score_threshold})"
        )

        try:
            fallback = pageindex_search(
                query,
                top_k=top_k,
            )

            if fallback:

                for item in fallback:
                    item.setdefault("source", "pageindex")

                return fallback[:top_k]

        except Exception as e:
            # Không để PageIndex làm crash pipeline
            print(f"PageIndex fallback failed: {e}")

            # Nếu fallback lỗi thì vẫn trả hybrid (hoặc [] nếu hybrid rỗng)
            return final_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":

    test_queries = [
        "What payment methods does Shopee support?",
        "How do I request a return or refund?",
        "What evidence do I need for a refund request?",
        "xyzabc123nonsense",
    ]

    for q in test_queries:

        print(f"\nQuery: {q}")
        print("-" * 60)

        results = retrieve(q, top_k=3)

        for i, r in enumerate(results, 1):
            print(
                f"{i}. "
                f"[{r['score']:.3f}] "
                f"[{r['source']}] "
                f"{r['content'][:80]}..."
            )