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
) -> list[dict]:
    """
    RAG retrieval flow (skeleton only):
    User question -> embedding -> similarity search -> retrieve top-k chunks

    This function assumes a FAISS index has already been built and persisted.
    """
    if not question or not question.strip():
        return []

    should_use_hybrid_search = config.USE_HYBRID_SEARCH if use_hybrid_search is None else use_hybrid_search

    if should_use_hybrid_search:
        hybrid_retriever = _get_hybrid_retriever()
        if hybrid_retriever is not None:
            try:
                return hybrid_retriever.retrieve(question=question, k=top_k)
            except FileNotFoundError:
                raise
            except Exception:
                logger.exception("Hybrid retrieval failed; falling back to vector search")

    # Step 1: Create query embedding from question.
    query_vector = create_query_embedding(question)

    # Step 2: Load vector store and run similarity search.
    index, documents = load_vectorstore()
    results = search_similar(index=index, query_vector=query_vector, top_k=top_k, documents=documents)

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
