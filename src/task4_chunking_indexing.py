"""Task 4 - split the standardized documents and index them in ChromaDB.

The public helpers in this module are also used by Tasks 5 and 6.  Expensive
objects (the embedding model and Chroma client) are therefore created lazily.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
CHROMA_DIR = PROJECT_DIR / "chroma_db"


# A chunk of 800 characters normally contains one or two short paragraphs,
# enough context for support questions without mixing too many subjects.
# 100 characters of overlap preserves sentences that cross a chunk boundary.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# BGE-M3 is multilingual and works well for both Vietnamese and English.  Its
# output has 1024 dimensions.  Normalized vectors plus Chroma's cosine metric
# make the returned distance directly convertible to cosine similarity.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# ChromaDB is local, persistent, and does not require a separate server.
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ecommerce_support_docs"


def _validate_chunk_config() -> None:
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero")
    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be in [0, CHUNK_SIZE)")


def load_documents() -> list[dict]:
    """Read every Markdown file under ``data/standardized``.

    Files are sorted to make chunk IDs and retrieval results reproducible.
    ``source_path`` disambiguates equal file names in different directories,
    while ``source`` remains convenient for displaying citations.
    """
    if not STANDARDIZED_DIR.exists():
        return []

    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "source_path": relative_path.as_posix(),
                    "type": doc_type,
                },
            }
        )
    return documents


def _best_boundary(text: str, start: int, hard_end: int) -> int:
    """Find a natural split point, falling back to the hard size limit."""
    if hard_end >= len(text):
        return len(text)

    # Do not create a very small chunk merely because an early separator was
    # found.  Separators are ordered from stronger to weaker.
    minimum = start + max(CHUNK_OVERLAP + 1, CHUNK_SIZE // 2)
    for separator in ("\n\n", "\n", ". ", "; ", ", ", " "):
        position = text.rfind(separator, minimum, hard_end)
        if position != -1:
            return position + len(separator)
    return hard_end


def _split_text(text: str) -> list[str]:
    """A dependency-free recursive-style, boundary-aware text splitter."""
    _validate_chunk_config()
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        hard_end = min(start + CHUNK_SIZE, text_length)
        end = _best_boundary(text, start, hard_end)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break

        next_start = max(0, end - CHUNK_OVERLAP)
        # Do not begin an overlapping chunk in the middle of a word.
        if next_start > 0 and not text[next_start - 1].isspace() and not text[next_start].isspace():
            while next_start < end and not text[next_start].isspace():
                next_start += 1
        # Skip boundary whitespace without consuming meaningful content.
        while next_start < end and text[next_start].isspace():
            next_start += 1
        if next_start <= start:  # Defensive guard against an invalid loop.
            next_start = end
        start = next_start
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents and attach a stable ``chunk_index`` to each result."""
    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    chunks: list[dict] = []
    for document in documents:
        if not isinstance(document, dict) or not isinstance(document.get("content"), str):
            raise ValueError("each document must contain a string 'content' field")
        metadata = document.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("document 'metadata' must be a dictionary")

        for chunk_index, chunk_text in enumerate(_split_text(document["content"])):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {**metadata, "chunk_index": chunk_index},
                }
            )
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    """Return the shared SentenceTransformer model used by Tasks 4 and 5."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # Give a concise, actionable error.
        raise RuntimeError(
            "sentence-transformers is required; run: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(EMBEDDING_MODEL)


def _encode(model: Any, texts: list[str]):
    """Encode normalized vectors, with compatibility for older model versions."""
    try:
        return model.encode(
            texts,
            show_progress_bar=len(texts) > 1,
            normalize_embeddings=True,
        )
    except TypeError:
        # Older/custom SentenceTransformer-compatible models may not accept
        # normalize_embeddings. Chroma still computes cosine distance safely.
        return model.encode(texts, show_progress_bar=len(texts) > 1)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach a JSON-serializable embedding to every chunk and return them."""
    if not chunks:
        return []
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("content"), str):
            raise ValueError("each chunk must contain a string 'content' field")

    embeddings = _encode(get_embedding_model(), [chunk["content"] for chunk in chunks])
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    if len(embeddings) != len(chunks):
        raise RuntimeError("embedding model returned an unexpected number of vectors")

    for chunk, embedding in zip(chunks, embeddings):
        vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        chunk["embedding"] = vector
    return chunks


@lru_cache(maxsize=1)
def get_chroma_client():
    """Return the shared persistent Chroma client."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("chromadb is required; run: pip install chromadb") from exc
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(*, create: bool = True):
    """Get the configured collection, optionally returning ``None`` if absent."""
    client = get_chroma_client()
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        # Chroma has changed the concrete not-found exception between releases.
        if exc.__class__.__name__ in {"NotFoundError", "InvalidCollectionException"}:
            return None
        message = str(exc).lower()
        if "does not exist" in message or "not found" in message:
            return None
        raise


def _chunk_id(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}
    source = metadata.get("source_path") or metadata.get("source") or "document"
    chunk_index = metadata.get("chunk_index", 0)
    return f"{source}::chunk::{chunk_index}"


def index_to_vectorstore(chunks: list[dict]):
    """Upsert embedded chunks into ChromaDB and return the collection."""
    if not chunks:
        return get_collection(create=True)

    for chunk in chunks:
        if "embedding" not in chunk:
            raise ValueError("all chunks must be embedded before indexing")
        if not isinstance(chunk.get("metadata", {}), dict):
            raise ValueError("chunk 'metadata' must be a dictionary")

    collection = get_collection(create=True)
    current_ids = {_chunk_id(chunk) for chunk in chunks}
    # Small batches avoid Chroma's maximum-batch-size limit on larger corpora.
    batch_size = 128
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[_chunk_id(chunk) for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[chunk.get("metadata") or {} for chunk in batch],
        )

    # Remove chunks left behind when a source document was deleted or became
    # shorter. This keeps repeated pipeline runs in sync with the filesystem.
    existing = collection.get(include=[])
    stale_ids = set(existing.get("ids") or []) - current_ids
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
    return collection


def run_pipeline() -> None:
    """Run the complete load -> chunk -> embed -> index pipeline."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    if not documents:
        print(f"No Markdown files found under {STANDARDIZED_DIR}")
        return

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")
    index_to_vectorstore(embedded_chunks)
    print("Indexed chunks to ChromaDB")


if __name__ == "__main__":
    run_pipeline()
