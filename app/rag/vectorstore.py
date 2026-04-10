"""Vector store stage for RAG using FAISS persistence."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app import config


INDEX_FILENAME = "faiss.index"
DOCUMENTS_FILENAME = "documents.json"
EMBEDDINGS_FILENAME = "embeddings.json"


def _resolve_store_dir(persist_dir: str | Path | None) -> Path:
    """Resolve the directory used to persist the vector store."""
    if persist_dir is None:
        return Path(config.VECTORSTORE_DIR)

    return Path(persist_dir)


def _to_normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
    """Convert vectors to a 2D float32 matrix and normalize for cosine search."""
    if not vectors:
        raise ValueError("vectors cannot be empty")

    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D list of embeddings")

    faiss.normalize_L2(matrix)
    return matrix


def save_embeddings_and_vectorstore(
    documents: list[dict],
    vectors: list[list[float]],
    persist_dir: str | Path | None = None,
) -> dict:
    """
    Save embeddings and a FAISS vector store to disk.

    Expected document shape:
    - {"text": "...", "metadata": {...}}

    This persists:
    - faiss.index: normalized vector index
    - documents.json: chunk text + metadata
    - embeddings.json: raw embedding vectors for inspection/debugging
    """
    if len(documents) != len(vectors):
        raise ValueError("documents and vectors must have the same length")

    store_dir = _resolve_store_dir(persist_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    vector_matrix = _to_normalized_matrix(vectors)
    index = faiss.IndexFlatIP(vector_matrix.shape[1])
    index.add(vector_matrix)

    faiss.write_index(index, str(store_dir / INDEX_FILENAME))

    serialized_documents = [
        {
            "text": document.get("text", ""),
            "metadata": document.get("metadata", {}),
        }
        for document in documents
    ]

    (store_dir / DOCUMENTS_FILENAME).write_text(
        json.dumps(serialized_documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (store_dir / EMBEDDINGS_FILENAME).write_text(
        json.dumps(vectors, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "store_dir": str(store_dir),
        "index_path": str(store_dir / INDEX_FILENAME),
        "documents_path": str(store_dir / DOCUMENTS_FILENAME),
        "embeddings_path": str(store_dir / EMBEDDINGS_FILENAME),
        "document_count": len(documents),
        "dimension": vector_matrix.shape[1],
    }


def build_faiss_index(
    vectors: list[list[float]],
    chunks: list[str | dict],
    persist_dir: str | Path | None = None,
):
    """
    Backward-compatible wrapper for building and saving a FAISS index.

    If chunks are strings, they are wrapped into document dictionaries.
    """
    documents: list[dict] = []

    for chunk in chunks:
        if isinstance(chunk, dict):
            documents.append(chunk)
        else:
            documents.append({"text": str(chunk), "metadata": {}})

    info = save_embeddings_and_vectorstore(documents, vectors, persist_dir=persist_dir)
    return info


def load_vectorstore(persist_dir: str | Path | None = None) -> tuple[faiss.Index, list[dict]]:
    """Load a persisted FAISS vector store and its documents metadata from disk."""
    store_dir = _resolve_store_dir(persist_dir)
    index_path = store_dir / INDEX_FILENAME
    documents_path = store_dir / DOCUMENTS_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    index = faiss.read_index(str(index_path))

    if documents_path.exists():
        documents = json.loads(documents_path.read_text(encoding="utf-8"))
    else:
        documents = []

    return index, documents


def search_similar(
    index: faiss.Index,
    query_vector: list[float],
    top_k: int,
    documents: list[dict] | None = None,
) -> list[dict]:
    """Run cosine similarity search and return top-k matching chunks."""
    if top_k <= 0:
        return []

    if not query_vector:
        raise ValueError("query_vector cannot be empty")

    query_matrix = np.asarray([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_matrix)

    scores, indices = index.search(query_matrix, top_k)

    results: list[dict] = []
    for score, position in zip(scores[0], indices[0]):
        if position < 0:
            continue

        document = documents[position] if documents and position < len(documents) else {}
        results.append(
            {
                "score": float(score),
                "text": document.get("text", ""),
                "metadata": document.get("metadata", {}),
            }
        )

    return results
