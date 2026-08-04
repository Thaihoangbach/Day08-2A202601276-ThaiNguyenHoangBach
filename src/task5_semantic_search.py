"""Task 5 - semantic (dense) search over the Task 4 Chroma collection."""

from __future__ import annotations

try:  # Support both ``python -m src.task5...`` and direct script execution.
    from .task4_chunking_indexing import get_collection, get_embedding_model
except ImportError:  # pragma: no cover - used only by direct CLI execution
    from task4_chunking_indexing import get_collection, get_embedding_model


def _validate_search_input(query: str, top_k: int) -> tuple[str, int]:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    return query.strip(), top_k


def _query_embedding(query: str) -> list[float]:
    model = get_embedding_model()
    try:
        embedding = model.encode(query, normalize_embeddings=True)
    except TypeError:
        embedding = model.encode(query)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    # Some compatible models return shape (1, dimension) for a single string.
    if embedding and isinstance(embedding[0], (list, tuple)):
        embedding = embedding[0]
    return [float(value) for value in embedding]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return the most similar indexed chunks, sorted by cosine similarity.

    An empty list is returned when the query is blank, ``top_k`` is not
    positive, or Task 4 has not created a collection yet.
    """
    query, top_k = _validate_search_input(query, top_k)
    if not query or top_k <= 0:
        return []

    # Check the collection before loading the large embedding model. When the
    # optional Chroma dependency is absent there cannot be an indexed corpus.
    try:
        collection = get_collection(create=False)
    except RuntimeError as exc:
        if "chromadb is required" not in str(exc).lower():
            raise
        return []
    if collection is None:
        return []
    count = int(collection.count())
    if count == 0:
        return []

    raw_results = collection.query(
        query_embeddings=[_query_embedding(query)],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = (raw_results.get("documents") or [[]])[0]
    metadatas = (raw_results.get("metadatas") or [[]])[0]
    distances = (raw_results.get("distances") or [[]])[0]

    results: list[dict] = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        if content is None or distance is None:
            continue
        # With hnsw:space=cosine, Chroma distance is 1 - cosine similarity.
        # Clamp to [0, 1] so downstream thresholds have a predictable scale.
        score = min(1.0, max(0.0, 1.0 - float(distance)))
        results.append(
            {
                "content": str(content),
                "score": float(round(score, 6)),
                "metadata": dict(metadata or {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    matches = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    if not matches:
        print("No indexed data. Run: python -m src.task4_chunking_indexing")
    for match in matches:
        print(f"[{match['score']:.3f}] {match['content'][:100]}...")
