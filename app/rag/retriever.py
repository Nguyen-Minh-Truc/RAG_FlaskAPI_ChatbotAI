"""Retriever stage for RAG."""

import logging
from functools import lru_cache
from typing import Any

from app import config
from app.rag.embeddings import create_query_embedding
from app.rag.vectorstore import load_vectorstore, search_similar

try:
    from hybrid_search import HybridSearchRetriever
except Exception:  # pragma: no cover - optional during bootstrap
    HybridSearchRetriever = None

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_hybrid_retriever() -> Any:
    """Build and cache the default hybrid retriever instance."""
    if HybridSearchRetriever is None:
        return None

    retriever = HybridSearchRetriever(top_k=config.TOP_K)
    return retriever.build()


def retrieve_top_k_chunks(
    question: str,
    top_k: int = 3,
    use_hybrid_search: bool | None = None,
    metadata_filters: dict | None = None,
) -> list[dict]:
    """
    RAG retrieval flow (skeleton only):
    User question -> embedding -> similarity search -> retrieve top-k chunks

    This function assumes a FAISS index has already been built and persisted.
    """
    if not question or not question.strip():
        return []

    should_use_hybrid_search = config.USE_HYBRID_SEARCH if use_hybrid_search is None else use_hybrid_search
    effective_filters = metadata_filters or {}

    def _metadata_matches(item: dict) -> bool:
        metadata = item.get("metadata", {}) or {}
        sources = effective_filters.get("sources") or []
        file_types = effective_filters.get("file_types") or []
        document_ids = effective_filters.get("document_ids") or []
        upload_date_from = effective_filters.get("upload_date_from")
        upload_date_to = effective_filters.get("upload_date_to")

        if sources and metadata.get("source") not in sources:
            return False

        if file_types and metadata.get("file_type") not in file_types:
            return False

        if document_ids and metadata.get("document_id") not in document_ids:
            return False

        upload_date = metadata.get("upload_date")
        if upload_date_from and (not upload_date or str(upload_date) < str(upload_date_from)):
            return False

        if upload_date_to and (not upload_date or str(upload_date) > str(upload_date_to)):
            return False

        return True

    has_filters = any(
        effective_filters.get(key)
        for key in ("sources", "file_types", "document_ids", "upload_date_from", "upload_date_to")
    )

    if should_use_hybrid_search:
        hybrid_retriever = _get_hybrid_retriever()
        if hybrid_retriever is not None:
            try:
                hybrid_retriever.build()
                hybrid_top_k = top_k if not has_filters else max(top_k * 4, top_k + 10)
                hybrid_results = hybrid_retriever.retrieve(question=question, k=hybrid_top_k)
                if has_filters:
                    filtered_hybrid = [item for item in hybrid_results if _metadata_matches(item)]
                    if len(filtered_hybrid) >= top_k:
                        return filtered_hybrid[:top_k]
                else:
                    return hybrid_results
            except FileNotFoundError:
                raise
            except Exception:
                logger.exception("Hybrid retrieval failed; falling back to vector search")

    # Step 1: Create query embedding from question.
    query_vector = create_query_embedding(question)

    # Step 2: Load vector store and run similarity search.
    index, documents = load_vectorstore()
    results = search_similar(
        index=index,
        query_vector=query_vector,
        top_k=top_k,
        documents=documents,
        metadata_filters=effective_filters,
    )

    # Step 3: Return top-k chunks with score/metadata.
    normalized_results: list[dict] = []
    for item in results:
        normalized_results.append(
            {
                "text": item.get("text", ""),
                "score": item.get("score", 0.0),
                "metadata": item.get("metadata", {}),
            }
        )

    # Simple logging for learning/debugging retrieval outputs.
    for idx, item in enumerate(normalized_results, start=1):
        preview = item.get("text", "")[:120]
        logger.info("Retrieved chunk %s | score=%s | preview=%s", idx, item.get("score", 0.0), preview)

    return normalized_results
