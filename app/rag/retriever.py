"""Retriever stage for RAG."""

import logging

from app.rag.embeddings import create_query_embedding
from app.rag.vectorstore import load_vectorstore, search_similar

logger = logging.getLogger(__name__)


def retrieve_top_k_chunks(question: str, top_k: int = 3) -> list[dict]:
    """
    RAG retrieval flow (skeleton only):
    User question -> embedding -> similarity search -> retrieve top-k chunks

    This function assumes a FAISS index has already been built and persisted.
    """
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
