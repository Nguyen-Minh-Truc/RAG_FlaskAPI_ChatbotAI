"""Vector store stage for RAG using FAISS persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    merge_existing: bool = False,
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

    all_documents = list(documents)
    all_vectors = [list(vector) for vector in vectors]

    if merge_existing:
        existing_embeddings_path = store_dir / EMBEDDINGS_FILENAME
        existing_documents_path = store_dir / DOCUMENTS_FILENAME

        existing_documents: list[dict] = []
        existing_vectors: list[list[float]] = []

        if existing_documents_path.exists():
            existing_documents = json.loads(existing_documents_path.read_text(encoding="utf-8"))

        if existing_embeddings_path.exists():
            existing_vectors = json.loads(existing_embeddings_path.read_text(encoding="utf-8"))

        if len(existing_documents) != len(existing_vectors):
            raise ValueError("existing documents and embeddings are inconsistent")

        all_documents = existing_documents + all_documents
        all_vectors = existing_vectors + all_vectors

    vector_matrix = _to_normalized_matrix(all_vectors)
    index = faiss.IndexFlatIP(vector_matrix.shape[1])
    index.add(vector_matrix)

    faiss.write_index(index, str(store_dir / INDEX_FILENAME))

    serialized_documents = [
        {
            "text": document.get("text", ""),
            "metadata": document.get("metadata", {}),
        }
        for document in all_documents
    ]

    (store_dir / DOCUMENTS_FILENAME).write_text(
        json.dumps(serialized_documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (store_dir / EMBEDDINGS_FILENAME).write_text(
        json.dumps(all_vectors, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "store_dir": str(store_dir),
        "index_path": str(store_dir / INDEX_FILENAME),
        "documents_path": str(store_dir / DOCUMENTS_FILENAME),
        "embeddings_path": str(store_dir / EMBEDDINGS_FILENAME),
        "document_count": len(all_documents),
        "added_count": len(documents),
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
    metadata_filters: dict | None = None,
) -> list[dict]:
    """Run cosine similarity search and return top-k matching chunks."""
    if top_k <= 0:
        return []

    if not query_vector:
        raise ValueError("query_vector cannot be empty")

    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        source_values = filters.get("sources") or []
        file_type_values = filters.get("file_types") or []
        document_id_values = filters.get("document_ids") or []
        upload_date_from = filters.get("upload_date_from")
        upload_date_to = filters.get("upload_date_to")

        source = metadata.get("source")
        file_type = metadata.get("file_type")
        document_id = metadata.get("document_id")
        upload_date = metadata.get("upload_date")

        if source_values and source not in source_values:
            return False

        if file_type_values and file_type not in file_type_values:
            return False

        if document_id_values and document_id not in document_id_values:
            return False

        if upload_date_from and (not upload_date or str(upload_date) < str(upload_date_from)):
            return False

        if upload_date_to and (not upload_date or str(upload_date) > str(upload_date_to)):
            return False

        return True

    query_matrix = np.asarray([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_matrix)

    effective_filters = metadata_filters or {}
    has_filters = any(
        effective_filters.get(key)
        for key in ("sources", "file_types", "document_ids", "upload_date_from", "upload_date_to")
    )
    search_k = top_k
    if has_filters:
        search_k = min(index.ntotal, max(top_k * 8, top_k + 20))

    scores, indices = index.search(query_matrix, search_k)

    results: list[dict] = []
    for score, position in zip(scores[0], indices[0]):
        if position < 0:
            continue

        document = documents[position] if documents and position < len(documents) else {}
        metadata = document.get("metadata", {})
        if has_filters and not _matches_filters(metadata, effective_filters):
            continue

        results.append(
            {
                "score": float(score),
                "text": document.get("text", ""),
                "metadata": metadata,
            }
        )

        if len(results) >= top_k:
            break

    return results
