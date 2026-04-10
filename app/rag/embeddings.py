"""Embedding stage for RAG."""

from sentence_transformers import SentenceTransformer

from app import config


def _get_embedding_model() -> SentenceTransformer:
    """Create the shared SentenceTransformer embedding model instance."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def create_embeddings(chunks: list[str]) -> list[list[float]]:
    """
    Convert text chunks into embedding vectors.

    This uses the same embedding model for both documents and user questions
    so similarity search is done in a consistent vector space.
    """
    if not chunks:
        return []

    embedding_model = _get_embedding_model()
    vectors = embedding_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=False)
    return vectors.tolist()


def create_query_embedding(question: str) -> list[float]:
    """Convert a user question into an embedding vector."""
    if not question.strip():
        raise ValueError("question cannot be empty")

    embedding_model = _get_embedding_model()
    vector = embedding_model.encode(question, convert_to_numpy=True, normalize_embeddings=False)
    return vector.tolist()
