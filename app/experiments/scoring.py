"""Answer-quality proxy scoring for chunk tuning experiments."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app import config


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


def _to_01(cosine_value: float) -> float:
    # Map cosine [-1, 1] to [0, 1] for easier aggregation.
    return (cosine_value + 1.0) / 2.0


def score_answer(question: str, answer: str, context_chunks: list[dict]) -> dict:
    """Compute a proxy answer-quality score without ground truth."""
    model = _get_model()

    answer_text = (answer or "").strip()
    if not answer_text:
        return {
            "relevance": 0.0,
            "groundedness": 0.0,
            "context_coverage": 0.0,
            "clarity": 0.0,
            "accuracy_proxy": 0.0,
        }

    q_vec = model.encode(question, convert_to_numpy=True)
    a_vec = model.encode(answer_text, convert_to_numpy=True)

    context_texts = [(item.get("text") or "").strip() for item in context_chunks]
    context_texts = [text for text in context_texts if text]

    relevance = _to_01(_cosine(q_vec, a_vec))

    if context_texts:
        c_vecs = model.encode(context_texts, convert_to_numpy=True)
        context_similarities = [_to_01(_cosine(a_vec, c_vec)) for c_vec in c_vecs]
        groundedness = max(context_similarities)
        top_n = min(2, len(context_similarities))
        context_coverage = float(np.mean(sorted(context_similarities, reverse=True)[:top_n]))
    else:
        groundedness = 0.0
        context_coverage = 0.0

    token_count = len(answer_text.split())
    punctuation_bonus = 0.08 if any(ch in answer_text for ch in ".,;:!?") else 0.0
    # Keep clarity simple: not too short and has sentence-like structure.
    clarity = min(1.0, (token_count / 40.0) + punctuation_bonus)

    accuracy_proxy = (
        0.4 * groundedness
        + 0.3 * relevance
        + 0.2 * context_coverage
        + 0.1 * clarity
    )

    return {
        "relevance": round(relevance, 4),
        "groundedness": round(groundedness, 4),
        "context_coverage": round(context_coverage, 4),
        "clarity": round(clarity, 4),
        "accuracy_proxy": round(float(accuracy_proxy), 4),
    }
