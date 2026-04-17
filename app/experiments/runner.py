"""Grid-search runner for chunk parameter experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path

from werkzeug.datastructures import FileStorage

from app import config
from app.experiments.scoring import score_answer
from app.llm.llm_service import generate_answer
from app.rag.chunker import split_into_chunks
from app.rag.embeddings import create_embeddings, create_query_embedding
from app.rag.history import _history_dir
from app.rag.loader import load_uploaded_document
from app.rag.vectorstore import load_vectorstore, save_embeddings_and_vectorstore, search_similar

CHUNK_SIZE_GRID = [500, 1000, 1500, 2000]
CHUNK_OVERLAP_GRID = [50, 100, 200]


def _collect_questions(max_questions: int) -> list[str]:
    """Collect distinct historical questions as evaluation prompts."""
    unique_questions: list[str] = []
    seen: set[str] = set()

    for path in sorted(_history_dir().glob("*.json"), key=lambda item: item.stem):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for turn in payload.get("turns", []):
            question = (turn.get("question") or "").strip()
            normalized = " ".join(question.split())
            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            unique_questions.append(question)
            if len(unique_questions) >= max_questions:
                return unique_questions

    return unique_questions


def _load_documents_from_path(file_path: str) -> list[dict]:
    """Load source documents using existing upload loader logic."""
    file_name = Path(file_path).name
    with open(file_path, "rb") as handle:
        uploaded_file = FileStorage(
            stream=handle,
            filename=file_name,
            content_type="application/octet-stream",
        )
        return load_uploaded_document(uploaded_file)


def _run_single_configuration(
    documents: list[dict],
    questions: list[str],
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> dict:
    run_start = time.perf_counter()

    ingest_start = time.perf_counter()
    chunks = split_into_chunks(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vectors = create_embeddings([item.get("text", "") for item in chunks])

    store_dir = Path(config.VECTORSTORE_DIR) / "experiments" / f"cs{chunk_size}_ov{chunk_overlap}"
    save_embeddings_and_vectorstore(documents=chunks, vectors=vectors, persist_dir=store_dir)
    index, indexed_docs = load_vectorstore(persist_dir=store_dir)
    ingest_seconds = time.perf_counter() - ingest_start

    question_results: list[dict] = []
    retrieval_scores: list[float] = []
    latencies: list[float] = []

    for question in questions:
        ask_start = time.perf_counter()
        query_vector = create_query_embedding(question)
        retrieved = search_similar(
            index=index,
            query_vector=query_vector,
            top_k=top_k,
            documents=indexed_docs,
        )

        context_chunks = [
            {
                "text": item.get("text", ""),
                "score": float(item.get("score", 0.0)),
                "metadata": item.get("metadata", {}),
            }
            for item in retrieved
        ]
        retrieval_scores.extend([item.get("score", 0.0) for item in context_chunks])

        answer = generate_answer(question=question, context_chunks=context_chunks)
        score = score_answer(question=question, answer=answer, context_chunks=context_chunks)
        latency = time.perf_counter() - ask_start
        latencies.append(latency)

        question_results.append(
            {
                "question": question,
                "latency_sec": round(latency, 4),
                "answer": answer,
                "score": score,
            }
        )

    def _avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    result = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents": len(documents),
        "chunks": len(chunks),
        "questions": len(questions),
        "ingest_time_sec": round(ingest_seconds, 4),
        "avg_latency_sec": _avg(latencies),
        "avg_retrieval_score": _avg(retrieval_scores),
        "avg_accuracy_proxy": _avg([item["score"]["accuracy_proxy"] for item in question_results]),
        "avg_relevance": _avg([item["score"]["relevance"] for item in question_results]),
        "avg_groundedness": _avg([item["score"]["groundedness"] for item in question_results]),
        "runtime_sec": round(time.perf_counter() - run_start, 4),
        "samples": question_results[:3],
        "question_results": question_results,
    }

    return result


def run_chunk_grid_experiment(
    source_file_path: str,
    max_questions: int = 20,
    top_k: int = 3,
) -> dict:
    """Run chunk grid experiment and return all configuration results."""
    source_path = Path(source_file_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_file_path}")

    questions = _collect_questions(max_questions=max_questions)
    if not questions:
        raise ValueError("No historical questions found. Ask a few questions first to build evaluation set.")

    documents = _load_documents_from_path(str(source_path))
    if not documents:
        raise ValueError("Failed to extract text from source file.")

    all_results: list[dict] = []

    for chunk_size in CHUNK_SIZE_GRID:
        for chunk_overlap in CHUNK_OVERLAP_GRID:
            if chunk_overlap >= chunk_size:
                continue
            all_results.append(
                _run_single_configuration(
                    documents=documents,
                    questions=questions,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                )
            )

    return {
        "source_file": str(source_path),
        "top_k": top_k,
        "question_count": len(questions),
        "questions": questions,
        "results": all_results,
    }
