"""Co-RAG pipeline built on top of the existing RAG stack."""

from app import config
from app.llm.llm_service import generate_corag_refined_answer
from app.rag.retriever import retrieve_top_k_chunks


DEFAULT_CORAG_ROUNDS = 3


def _dedupe_and_rank(chunks: list[dict], limit: int) -> list[dict]:
    """Rank by score and remove duplicate chunk text."""
    ranked = sorted(chunks, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    seen_texts: set[str] = set()
    result: list[dict] = []

    for item in ranked:
        text = (item.get("text") or "").strip()
        if not text or text in seen_texts:
            continue

        seen_texts.add(text)
        result.append(item)
        if len(result) >= limit:
            break

    return result


def generate_corag_answer(
    question: str,
    base_chunks: list[dict] | None = None,
    base_answer: str = "",
    rounds: int = DEFAULT_CORAG_ROUNDS,
    memory_turns: list[dict] | None = None,
    original_question: str | None = None,
    use_hybrid_search: bool | None = None,
    metadata_filters: dict | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """
    Generate answer with iterative corrective retrieval and per-round trace.

    Each round:
    - retrieve additional chunks with increased top-k
    - merge with current context
    - dedupe and rerank context
    - generate intermediate answer
    """
    if not question.strip():
        raise ValueError("question cannot be empty")

    if rounds <= 0:
        raise ValueError("rounds must be greater than 0")

    requested_rounds = rounds
    rounds = min(rounds, 6)

    working_context: list[dict] = list(base_chunks or [])
    trace: list[dict] = []
    latest_answer = base_answer.strip()

    base_top_k = max(config.TOP_K, 1)

    for round_no in range(1, rounds + 1):
        # CORAG starts higher than base RAG to differentiate from simple RAG approach
        # Round 1: base_top_k * 2, Round 2: base_top_k * 2 + 2, Round 3: base_top_k * 2 + 4
        round_top_k = base_top_k * 2 + (round_no - 1) * 2

        retrieved_chunks = retrieve_top_k_chunks(
            question=question,
            top_k=round_top_k,
            use_hybrid_search=use_hybrid_search,
            metadata_filters=metadata_filters,
        )

        merged: list[dict] = []
        merged.extend(working_context)
        merged.extend(retrieved_chunks)

        context_limit = max(base_top_k * 2, round_top_k)
        working_context = _dedupe_and_rank(merged, limit=context_limit)

        latest_answer = generate_corag_refined_answer(
            question=question,
            context_chunks=working_context,
            previous_answer=latest_answer,
            round_no=round_no,
            rounds=rounds,
            memory_turns=memory_turns,
            original_question=original_question,
        )

        trace.append(
            {
                "round": round_no,
                "top_k": round_top_k,
                "retrieved_count": len(retrieved_chunks),
                "context_count": len(working_context),
                "requested_rounds": requested_rounds,
                "answer_preview": latest_answer[:240],
            }
        )

    return latest_answer, working_context, trace
