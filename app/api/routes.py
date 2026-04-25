"""REST API routes for RAG learning skeleton."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, request
from werkzeug.datastructures import FileStorage

from app import config
from app.corag.pipeline import DEFAULT_CORAG_ROUNDS, generate_corag_answer
from app.rag.chunker import split_into_chunks
from app.rag.embeddings import create_embeddings
from app.rag.history import (
    append_conversation_turn,
    append_uploaded_documents,
    create_conversation,
    delete_all_conversations,
    delete_conversation,
    get_recent_turns,
    list_conversations,
    load_conversation_history,
)
from app.rag.loader import SUPPORTED_UPLOAD_EXTENSIONS, load_uploaded_document
from app.rag.retriever import retrieve_top_k_chunks
from app.rag.vectorstore import save_embeddings_and_vectorstore
from app.llm.llm_service import generate_answer, rewrite_followup_question
from app.api.response import error_response, success_response

api_bp = Blueprint("api", __name__)

MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 5000
MIN_CHUNK_OVERLAP = 0
MAX_CHUNK_OVERLAP = 2000
REPORTS_DIR = Path(config.BASE_DIR) / "storage" / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_source_references(context_chunks: list[dict]) -> list[dict]:
    references: list[dict] = []
    for idx, chunk in enumerate(context_chunks, start=1):
        metadata = chunk.get("metadata", {}) or {}
        references.append(
            {
                "index": idx,
                "source": metadata.get("source", "unknown"),
                "document_id": metadata.get("document_id"),
                "upload_date": metadata.get("upload_date"),
                "file_type": metadata.get("file_type"),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "score": float(chunk.get("score", 0.0)),
            }
        )
    return references


def _build_document_source_summary(context_chunks: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for chunk in context_chunks:
        metadata = chunk.get("metadata", {}) or {}
        source = str(metadata.get("source") or "unknown")
        document_id = str(metadata.get("document_id") or source)
        key = f"{document_id}:{source}"
        if key not in grouped:
            grouped[key] = {
                "document_id": metadata.get("document_id"),
                "source": source,
                "file_type": metadata.get("file_type"),
                "upload_date": metadata.get("upload_date"),
                "chunk_count": 0,
                "total_score": 0.0,
                "pages": set(),
            }
        g = grouped[key]
        g["chunk_count"] += 1
        g["total_score"] += float(chunk.get("score", 0.0))
        page = metadata.get("page")
        if isinstance(page, int):
            g["pages"].add(page)

    summary: list[dict] = []
    total_chunks = sum(item["chunk_count"] for item in grouped.values())
    for item in grouped.values():
        chunk_count = int(item["chunk_count"])
        avg_score = (item["total_score"] / chunk_count) if chunk_count > 0 else 0.0
        contribution_pct = (chunk_count / total_chunks * 100.0) if total_chunks > 0 else 0.0
        summary.append(
            {
                "document_id": item["document_id"],
                "source": item["source"],
                "file_type": item["file_type"],
                "upload_date": item["upload_date"],
                "chunk_count": chunk_count,
                "avg_score": round(avg_score, 4),
                "contribution_pct": round(contribution_pct, 2),
                "pages": sorted(item["pages"]),
            }
        )
    summary.sort(key=lambda x: x.get("chunk_count", 0), reverse=True)
    return summary


def _normalize_metadata_filters(payload_filters: dict | None) -> dict:
    payload_filters = payload_filters or {}
    if not isinstance(payload_filters, dict):
        raise ValueError("Field 'metadata_filters' must be an object.")

    def _as_list_string(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    return {
        "sources": _as_list_string(payload_filters.get("sources")),
        "file_types": [item.lower() for item in _as_list_string(payload_filters.get("file_types"))],
        "document_ids": _as_list_string(payload_filters.get("document_ids")),
        "upload_date_from": (
            str(payload_filters.get("upload_date_from")).strip()
            if payload_filters.get("upload_date_from") is not None
            else ""
        ) or None,
        "upload_date_to": (
            str(payload_filters.get("upload_date_to")).strip()
            if payload_filters.get("upload_date_to") is not None
            else ""
        ) or None,
    }


def _derive_documents_from_history(history: dict) -> list[dict]:
    documents: list[dict] = list(history.get("uploaded_documents", []))
    if documents:
        return documents
    seen_keys: set[str] = set()
    for turn in history.get("turns", []):
        for chunk in turn.get("context", []):
            metadata = chunk.get("metadata", {}) or {}
            source = str(metadata.get("source") or "unknown")
            document_id = str(metadata.get("document_id") or source)
            key = f"{document_id}:{source}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            documents.append(
                {
                    "document_id": metadata.get("document_id") or document_id,
                    "source": source,
                    "upload_date": metadata.get("upload_date"),
                    "file_type": metadata.get("file_type") or Path(source).suffix.lower().lstrip("."),
                    "content_type": metadata.get("content_type"),
                }
            )
    return documents


def _coerce_report_questions(raw_questions, history_turns: list[dict], max_questions: int) -> list[str]:
    if isinstance(raw_questions, list):
        resolved = [str(item).strip() for item in raw_questions if str(item).strip()]
        if resolved:
            return resolved[:max_questions]
    deduped: list[str] = []
    seen: set[str] = set()
    for turn in history_turns:
        question = str(turn.get("question") or "").strip()
        if not question or question in seen:
            continue
        seen.add(question)
        deduped.append(question)
        if len(deduped) >= max_questions:
            break
    return deduped


def _persist_report(report_payload: dict) -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = REPORTS_DIR / f"rag_corag_document_report_{timestamp}.json"
    file_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(file_path)


def _retrieve_context_chunks(
    resolved_question: str,
    use_hybrid_search: bool | None,
    metadata_filters: dict | None = None,
):
    try:
        return retrieve_top_k_chunks(
            question=resolved_question,
            top_k=config.TOP_K,
            use_hybrid_search=use_hybrid_search,
            metadata_filters=metadata_filters,
        ), None
    except FileNotFoundError:
        return None, error_response(
            "Vector store not found. Please upload a file first via POST /api/upload.", 400
        )
    except Exception as exc:
        return None, error_response("Retrieval failed", 500, details={"detail": str(exc)})


def _persist_turn(
    conversation_id: str,
    question: str,
    resolved_question: str,
    memory_turns: list[dict],
    context: list[dict],
    rag_answer: str,
    corag_answer: str,
    mode: str,
    metadata_filters: dict,
    source_summary: list[dict],
):
    try:
        turn = append_conversation_turn(
            conversation_id=conversation_id,
            question=question,
            answer=rag_answer,
            corag_answer=corag_answer,
            context=context,
            memory_context=memory_turns,
            resolved_question=resolved_question,
            mode=mode,
            metadata_filters=metadata_filters,
            source_summary=source_summary,
        )
        return turn, None
    except FileNotFoundError:
        return None, error_response(
            "Conversation not found. Please upload a file first to create a conversation.", 400
        )
    except Exception as exc:
        return None, error_response("Failed to persist conversation history", 500, details={"detail": str(exc)})


def _validate_upload_files() -> tuple[list[FileStorage] | None, tuple | None]:
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        single_file = request.files.get("file")
        if single_file is not None:
            uploaded_files = [single_file]
    if not uploaded_files:
        return None, error_response("Field 'file' or 'files' is required.", 400)

    valid_files: list[FileStorage] = []
    allowed = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
    for uploaded_file in uploaded_files:
        if uploaded_file is None:
            continue
        if not uploaded_file.filename:
            return None, error_response("Filename is required for all uploaded files.", 400)
        extension = Path(uploaded_file.filename).suffix.lower()
        if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
            return None, error_response(f"Unsupported file type. Allowed: {allowed}", 400)
        valid_files.append(uploaded_file)

    if not valid_files:
        return None, error_response("No valid files found in upload payload.", 400)
    return valid_files, None


def _parse_upload_chunk_params() -> tuple[int, int] | tuple[None, tuple]:
    raw_chunk_size = request.form.get("chunk_size")
    raw_chunk_overlap = request.form.get("chunk_overlap")
    chunk_size = config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP

    if raw_chunk_size is not None and raw_chunk_size.strip() != "":
        try:
            chunk_size = int(raw_chunk_size)
        except ValueError:
            return None, error_response("Field 'chunk_size' must be an integer.", 400)

    if raw_chunk_overlap is not None and raw_chunk_overlap.strip() != "":
        try:
            chunk_overlap = int(raw_chunk_overlap)
        except ValueError:
            return None, error_response("Field 'chunk_overlap' must be an integer.", 400)

    if not (MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE):
        return None, error_response(
            f"Field 'chunk_size' must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}.", 400
        )
    if not (MIN_CHUNK_OVERLAP <= chunk_overlap <= MAX_CHUNK_OVERLAP):
        return None, error_response(
            f"Field 'chunk_overlap' must be between {MIN_CHUNK_OVERLAP} and {MAX_CHUNK_OVERLAP}.", 400
        )
    if chunk_overlap >= chunk_size:
        return None, error_response("Field 'chunk_overlap' must be smaller than 'chunk_size'.", 400)

    return (chunk_size, chunk_overlap), None


def _prepare_question_context(payload: dict, default_corag_rounds: int):
    question = payload.get("question")
    conversation_id = payload.get("conversation_id")
    corag_rounds = payload.get("corag_rounds", default_corag_rounds)
    use_hybrid_search = payload.get("use_hybrid_search")
    raw_metadata_filters = payload.get("metadata_filters")

    if not isinstance(question, str) or not question.strip():
        return None, error_response("Field 'question' is required and must be a non-empty string.", 400)
    if not isinstance(corag_rounds, int):
        return None, error_response("Field 'corag_rounds' must be an integer.", 400)
    if corag_rounds <= 0:
        return None, error_response("Field 'corag_rounds' must be greater than 0.", 400)
    if use_hybrid_search is not None and not isinstance(use_hybrid_search, bool):
        return None, error_response("Field 'use_hybrid_search' must be a boolean when provided.", 400)

    try:
        metadata_filters = _normalize_metadata_filters(raw_metadata_filters)
    except ValueError as exc:
        return None, error_response(str(exc), 400)

    question = question.strip()

    if isinstance(conversation_id, str) and conversation_id.strip():
        conversation_id = conversation_id.strip()
    else:
        try:
            conversations = list_conversations()
        except Exception as exc:
            return None, error_response("Failed to list conversations", 500, details={"detail": str(exc)})
        if not conversations:
            return None, error_response(
                "No conversation found. Please upload a file first via POST /api/upload.", 400
            )
        conversation_id = str(conversations[0].get("conversation_id"))

    try:
        memory_turns = get_recent_turns(conversation_id, limit=config.CONVERSATION_MEMORY_TURNS)
    except FileNotFoundError:
        memory_turns = []
    except Exception as exc:
        return None, error_response("Failed to load conversation memory", 500, details={"detail": str(exc)})

    resolved_question = question
    if memory_turns:
        try:
            resolved_question = rewrite_followup_question(question=question, memory_turns=memory_turns)
        except Exception:
            resolved_question = question

    return {
        "question": question,
        "conversation_id": conversation_id,
        "resolved_question": resolved_question,
        "memory_turns": memory_turns,
        "corag_rounds": corag_rounds,
        "use_hybrid_search": use_hybrid_search,
        "metadata_filters": metadata_filters,
    }, None


def _payload_or_error(default_corag_rounds: int):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, error_response("Invalid JSON body.", 400)
    prepared, validation_error = _prepare_question_context(payload, default_corag_rounds=default_corag_rounds)
    if validation_error:
        return None, validation_error
    return prepared, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@api_bp.get("/api/health")
def health_check():
    return success_response(data={"status": "ok"}, message="Service is healthy")


@api_bp.get("/api/conversations")
def get_conversations():
    try:
        conversations = list_conversations()
    except Exception as exc:
        return error_response("Failed to list conversations", 500, details={"detail": str(exc)})
    return success_response(data=conversations, message="Conversation list fetched successfully")


@api_bp.get("/api/conversations/<conversation_id>")
def get_conversation_detail(conversation_id: str):
    try:
        history = load_conversation_history(conversation_id=conversation_id)
    except FileNotFoundError:
        return error_response("Conversation not found.", 404)
    except Exception as exc:
        return error_response("Failed to load conversation", 500, details={"detail": str(exc)})
    return success_response(data=history, message="Conversation detail fetched successfully")


@api_bp.get("/api/conversations/<conversation_id>/documents")
def get_conversation_documents(conversation_id: str):
    try:
        history = load_conversation_history(conversation_id=conversation_id)
    except FileNotFoundError:
        return error_response("Conversation not found.", 404)
    except Exception as exc:
        return error_response("Failed to load conversation documents", 500, details={"detail": str(exc)})

    try:
        documents = _derive_documents_from_history(history)
    except Exception as exc:
        return error_response("Failed to derive documents from history", 500, details={"detail": str(exc)})

    return success_response(
        data={
            "conversation_id": conversation_id,
            "documents": documents,
            "document_count": len(documents),
        },
        message="Conversation documents fetched successfully",
    )


@api_bp.delete("/api/conversations/<conversation_id>")
def remove_conversation(conversation_id: str):
    try:
        deleted = delete_conversation(conversation_id)
    except Exception as exc:
        return error_response("Failed to delete conversation", 500, details={"detail": str(exc)})
    if not deleted:
        return error_response("Conversation not found.", 404)
    return success_response(data={"conversation_id": conversation_id}, message="Conversation deleted successfully")


@api_bp.delete("/api/conversations")
def remove_all_conversations():
    try:
        deleted_count = delete_all_conversations()
    except Exception as exc:
        return error_response("Failed to delete all conversations", 500, details={"detail": str(exc)})
    return success_response(data={"deleted": deleted_count}, message="All conversations deleted successfully")


@api_bp.post("/api/upload")
def ingest_pdf():
    uploaded_files, validation_error = _validate_upload_files()
    if validation_error:
        return validation_error

    chunk_params, chunk_error = _parse_upload_chunk_params()
    if chunk_error:
        return chunk_error

    chunk_size, chunk_overlap = chunk_params
    requested_conversation_id = (request.form.get("conversation_id") or "").strip()

    if requested_conversation_id:
        try:
            conversation = load_conversation_history(requested_conversation_id)
        except FileNotFoundError:
            return error_response("Conversation not found for append upload.", 400)
        except Exception as exc:
            return error_response("Failed to load conversation", 500, details={"detail": str(exc)})
        conversation_id = requested_conversation_id
        merge_existing = True
        base_upload_filename = str(conversation.get("upload_filename") or (uploaded_files[0].filename or "uploaded"))
    else:
        base_upload_filename = uploaded_files[0].filename or "uploaded"
        try:
            conversation_id = create_conversation(base_upload_filename)
        except Exception as exc:
            return error_response("Failed to create conversation", 500, details={"detail": str(exc)})
        merge_existing = False

    all_documents: list[dict] = []
    all_chunks: list[dict] = []
    uploaded_documents: list[dict] = []
    parse_errors: list[dict] = []
    uploaded_at = datetime.now(timezone.utc).isoformat()

    for uploaded_file in uploaded_files:
        extension = Path(uploaded_file.filename or "").suffix.lower() or "unknown"
        document_id = str(uuid4())
        file_type = extension.lstrip(".") if extension.startswith(".") else extension
        document_metadata = {
            "document_id": document_id,
            "upload_date": uploaded_at,
            "file_type": file_type,
        }

        try:
            documents = load_uploaded_document(uploaded_file, document_metadata=document_metadata)
        except (ValueError, Exception) as exc:
            parse_errors.append({"filename": uploaded_file.filename, "extension": extension, "detail": str(exc)})
            continue

        if not documents:
            parse_errors.append({"filename": uploaded_file.filename, "extension": extension, "detail": "No readable text extracted."})
            continue

        try:
            chunks = split_into_chunks(documents=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception as exc:
            parse_errors.append({"filename": uploaded_file.filename, "extension": extension, "detail": f"Chunking failed: {exc}"})
            continue

        all_documents.extend(documents)
        all_chunks.extend(chunks)
        uploaded_documents.append(
            {
                "document_id": document_id,
                "source": uploaded_file.filename,
                "upload_date": uploaded_at,
                "file_type": file_type,
                "content_type": uploaded_file.content_type,
                "document_parts": len(documents),
                "chunk_count": len(chunks),
            }
        )

    if not all_documents or not all_chunks:
        return error_response(
            "Failed to parse uploaded files",
            400,
            details={
                "detail": "No readable text was extracted from the uploaded files.",
                "errors": parse_errors,
                "hint": "Try text-based PDF/DOCX files or run OCR for image-only documents.",
            },
        )

    try:
        chunk_texts = [chunk.get("text", "") for chunk in all_chunks]
        vectors = create_embeddings(chunk_texts)
    except Exception as exc:
        return error_response("Embedding creation failed", 500, details={"detail": str(exc)})

    try:
        store_info = save_embeddings_and_vectorstore(
            documents=all_chunks,
            vectors=vectors,
            persist_dir=config.VECTORSTORE_DIR,
            merge_existing=merge_existing,
        )
    except Exception as exc:
        return error_response("Failed to save to vector store", 500, details={"detail": str(exc)})

    try:
        persisted_uploaded_documents = append_uploaded_documents(
            conversation_id=conversation_id,
            uploaded_documents=uploaded_documents,
        )
    except Exception as exc:
        return error_response("Failed to persist upload metadata", 500, details={"detail": str(exc)})

    return success_response(
        data={
            "conversation_id": conversation_id,
            "documents": uploaded_documents,
            "document_count": len(uploaded_documents),
            "segments": len(all_documents),
            "chunks": len(all_chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "merge_existing": merge_existing,
            "parse_errors": parse_errors,
            "uploaded_documents": persisted_uploaded_documents,
            "vectorstore": {
                "total_chunks": store_info.get("document_count", 0),
                "added_chunks": store_info.get("added_count", 0),
            },
        },
        message="Document ingestion completed successfully",
    )


@api_bp.post("/api/rag-ask")
def ask_question_rag_only():
    prepared, payload_error = _payload_or_error(default_corag_rounds=DEFAULT_CORAG_ROUNDS)
    if payload_error:
        return payload_error

    context_chunks, retrieval_error = _retrieve_context_chunks(
        resolved_question=prepared["resolved_question"],
        use_hybrid_search=prepared["use_hybrid_search"],
        metadata_filters=prepared["metadata_filters"],
    )
    if retrieval_error:
        return retrieval_error

    try:
        rag_answer = generate_answer(question=prepared["resolved_question"], context_chunks=context_chunks)
    except Exception as exc:
        return error_response("RAG generation failed", 500, details={"detail": str(exc)})

    turn, persist_error = _persist_turn(
        conversation_id=prepared["conversation_id"],
        question=prepared["question"],
        resolved_question=prepared["resolved_question"],
        memory_turns=prepared["memory_turns"],
        context=context_chunks,
        rag_answer=rag_answer,
        corag_answer="",
        mode="RAG",
        metadata_filters=prepared["metadata_filters"],
        source_summary=_build_document_source_summary(context_chunks),
    )
    if persist_error:
        return persist_error

    return success_response(
        data={
            "conversation_id": prepared["conversation_id"],
            "turn_id": turn.get("turn_id"),
            "original_question": prepared["question"],
            "resolved_question": prepared["resolved_question"],
            "use_hybrid_search": config.USE_HYBRID_SEARCH if prepared["use_hybrid_search"] is None else prepared["use_hybrid_search"],
            "rag_answer": rag_answer,
            "corag_answer": "",
            "corag_trace": [],
            "context": context_chunks,
            "source_references": _build_source_references(context_chunks),
            "source_summary": _build_document_source_summary(context_chunks),
            "memory_turns": prepared["memory_turns"],
            "metadata_filters": prepared["metadata_filters"],
            "mode": "RAG",
        },
        message="RAG answer generated successfully",
    )


@api_bp.post("/api/corag-ask")
def ask_question_corag_only():
    prepared, payload_error = _payload_or_error(default_corag_rounds=DEFAULT_CORAG_ROUNDS)
    if payload_error:
        return payload_error

    context_chunks, retrieval_error = _retrieve_context_chunks(
        resolved_question=prepared["resolved_question"],
        use_hybrid_search=prepared["use_hybrid_search"],
        metadata_filters=prepared["metadata_filters"],
    )
    if retrieval_error:
        return retrieval_error

    try:
        base_rag_answer = generate_answer(question=prepared["resolved_question"], context_chunks=context_chunks)
    except Exception as exc:
        return error_response("RAG draft generation failed", 500, details={"detail": str(exc)})

    try:
        corag_answer, corag_context, corag_trace = generate_corag_answer(
            question=prepared["resolved_question"],
            base_chunks=context_chunks,
            base_answer=base_rag_answer,
            rounds=prepared["corag_rounds"],
            memory_turns=prepared["memory_turns"],
            original_question=prepared["question"],
            use_hybrid_search=prepared["use_hybrid_search"],
            metadata_filters=prepared["metadata_filters"],
        )
    except Exception as exc:
        return error_response("Co-RAG generation failed", 500, details={"detail": str(exc)})

    turn, persist_error = _persist_turn(
        conversation_id=prepared["conversation_id"],
        question=prepared["question"],
        resolved_question=prepared["resolved_question"],
        memory_turns=prepared["memory_turns"],
        context=corag_context,
        rag_answer=base_rag_answer,
        corag_answer=corag_answer,
        mode="Co-RAG",
        metadata_filters=prepared["metadata_filters"],
        source_summary=_build_document_source_summary(corag_context),
    )
    if persist_error:
        return persist_error

    return success_response(
        data={
            "conversation_id": prepared["conversation_id"],
            "turn_id": turn.get("turn_id"),
            "original_question": prepared["question"],
            "resolved_question": prepared["resolved_question"],
            "use_hybrid_search": config.USE_HYBRID_SEARCH if prepared["use_hybrid_search"] is None else prepared["use_hybrid_search"],
            "rag_answer": "",
            "corag_answer": corag_answer,
            "corag_trace": corag_trace,
            "context": corag_context,
            "source_references": _build_source_references(corag_context),
            "source_summary": _build_document_source_summary(corag_context),
            "memory_turns": prepared["memory_turns"],
            "metadata_filters": prepared["metadata_filters"],
            "mode": "Co-RAG",
        },
        message="Co-RAG answer generated successfully",
    )


@api_bp.post("/api/ask")
def ask_question():
    prepared, payload_error = _payload_or_error(default_corag_rounds=DEFAULT_CORAG_ROUNDS)
    if payload_error:
        return payload_error

    context_chunks, retrieval_error = _retrieve_context_chunks(
        resolved_question=prepared["resolved_question"],
        use_hybrid_search=prepared["use_hybrid_search"],
        metadata_filters=prepared["metadata_filters"],
    )
    if retrieval_error:
        return retrieval_error

    def _run_rag() -> str:
        return generate_answer(question=prepared["resolved_question"], context_chunks=context_chunks)

    def _run_corag() -> tuple[str, list[dict], list[dict]]:
        return generate_corag_answer(
            question=prepared["resolved_question"],
            base_chunks=context_chunks,
            base_answer="",
            rounds=prepared["corag_rounds"],
            memory_turns=prepared["memory_turns"],
            original_question=prepared["question"],
            use_hybrid_search=prepared["use_hybrid_search"],
            metadata_filters=prepared["metadata_filters"],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        rag_future = executor.submit(_run_rag)
        corag_future = executor.submit(_run_corag)

        try:
            rag_answer = rag_future.result()
        except Exception as exc:
            return error_response("RAG generation failed", 500, details={"detail": str(exc)})

        try:
            corag_answer, _, corag_trace = corag_future.result()
        except Exception as exc:
            return error_response("Co-RAG generation failed", 500, details={"detail": str(exc)})

    turn, persist_error = _persist_turn(
        conversation_id=prepared["conversation_id"],
        question=prepared["question"],
        resolved_question=prepared["resolved_question"],
        memory_turns=prepared["memory_turns"],
        context=context_chunks,
        rag_answer=rag_answer,
        corag_answer=corag_answer,
        mode="Compare",
        metadata_filters=prepared["metadata_filters"],
        source_summary=_build_document_source_summary(context_chunks),
    )
    if persist_error:
        return persist_error

    return success_response(
        data={
            "conversation_id": prepared["conversation_id"],
            "turn_id": turn.get("turn_id"),
            "original_question": prepared["question"],
            "resolved_question": prepared["resolved_question"],
            "use_hybrid_search": config.USE_HYBRID_SEARCH if prepared["use_hybrid_search"] is None else prepared["use_hybrid_search"],
            "rag_answer": rag_answer,
            "corag_answer": corag_answer,
            "corag_trace": corag_trace,
            "context": context_chunks,
            "source_references": _build_source_references(context_chunks),
            "source_summary": _build_document_source_summary(context_chunks),
            "memory_turns": prepared["memory_turns"],
            "metadata_filters": prepared["metadata_filters"],
            "mode": "Compare",
        },
        message="RAG and Co-RAG answers generated successfully",
    )


@api_bp.post("/api/reports/rag-corag-by-document")
def export_rag_corag_report_by_document():
    payload = request.get_json(silent=True)
    if payload is not None and not isinstance(payload, dict):
        return error_response("Invalid JSON body.", 400)

    payload = payload or {}
    requested_conversation_id = str(payload.get("conversation_id") or "").strip()
    corag_rounds = payload.get("corag_rounds", DEFAULT_CORAG_ROUNDS)
    max_questions = payload.get("max_questions", 8)
    use_hybrid_search = payload.get("use_hybrid_search")

    if not isinstance(corag_rounds, int) or corag_rounds <= 0:
        return error_response("Field 'corag_rounds' must be an integer greater than 0.", 400)
    if not isinstance(max_questions, int) or max_questions <= 0:
        return error_response("Field 'max_questions' must be an integer greater than 0.", 400)
    if use_hybrid_search is not None and not isinstance(use_hybrid_search, bool):
        return error_response("Field 'use_hybrid_search' must be a boolean when provided.", 400)

    try:
        base_filters = _normalize_metadata_filters(payload.get("metadata_filters"))
    except ValueError as exc:
        return error_response(str(exc), 400)

    if requested_conversation_id:
        conversation_id = requested_conversation_id
    else:
        try:
            conversations = list_conversations()
        except Exception as exc:
            return error_response("Failed to list conversations", 500, details={"detail": str(exc)})
        if not conversations:
            return error_response("No conversation found. Please upload a file first.", 400)
        conversation_id = str(conversations[0].get("conversation_id"))

    try:
        history = load_conversation_history(conversation_id=conversation_id)
    except FileNotFoundError:
        return error_response("Conversation not found.", 404)
    except Exception as exc:
        return error_response("Failed to load conversation", 500, details={"detail": str(exc)})

    try:
        documents = _derive_documents_from_history(history)
    except Exception as exc:
        return error_response("Failed to derive documents from history", 500, details={"detail": str(exc)})

    if not documents:
        return error_response("No document metadata found for this conversation.", 400)

    # Filter documents
    filtered_documents: list[dict] = []
    for document in documents:
        source = str(document.get("source") or "")
        file_type = str(document.get("file_type") or "").lower()
        document_id = str(document.get("document_id") or "")
        upload_date = str(document.get("upload_date") or "")
        if base_filters.get("sources") and source not in base_filters["sources"]:
            continue
        if base_filters.get("file_types") and file_type not in base_filters["file_types"]:
            continue
        if base_filters.get("document_ids") and document_id not in base_filters["document_ids"]:
            continue
        if base_filters.get("upload_date_from") and (not upload_date or upload_date < base_filters["upload_date_from"]):
            continue
        if base_filters.get("upload_date_to") and (not upload_date or upload_date > base_filters["upload_date_to"]):
            continue
        filtered_documents.append(document)

    if not filtered_documents:
        return error_response("No documents matched the provided metadata_filters.", 400)

    questions = _coerce_report_questions(
        raw_questions=payload.get("questions"),
        history_turns=history.get("turns", []),
        max_questions=max_questions,
    )
    if not questions:
        return error_response("No questions available to build report. Please provide 'questions' in payload.", 400)

    report_rows: list[dict] = []

    for document in filtered_documents:
        document_id = str(document.get("document_id") or "")
        source = str(document.get("source") or "unknown")
        document_filter = dict(base_filters)
        document_filter["document_ids"] = [document_id] if document_id else []
        if not document_filter["document_ids"]:
            document_filter["sources"] = [source]

        per_question_results: list[dict] = []
        rag_latencies: list[float] = []
        corag_latencies: list[float] = []
        similarities: list[float] = []
        rag_scores: list[float] = []
        corag_scores: list[float] = []

        for question in questions:
            retrieval_start = time.perf_counter()
            context_chunks, retrieval_error = _retrieve_context_chunks(
                resolved_question=question,
                use_hybrid_search=use_hybrid_search,
                metadata_filters=document_filter,
            )
            retrieval_latency = round(time.perf_counter() - retrieval_start, 4)

            if retrieval_error is not None or not context_chunks:
                per_question_results.append(
                    {
                        "question": question,
                        "retrieval_latency_sec": retrieval_latency,
                        "rag_latency_sec": None,
                        "corag_latency_sec": None,
                        "similarity_ratio": None,
                        "rag_answer": "",
                        "corag_answer": "",
                        "context_count": 0,
                        "source_summary": [],
                        "error": "No context retrieved for this filter.",
                    }
                )
                continue

            try:
                rag_start = time.perf_counter()
                rag_answer = generate_answer(question=question, context_chunks=context_chunks)
                rag_latency = round(time.perf_counter() - rag_start, 4)
            except Exception as exc:
                per_question_results.append(
                    {
                        "question": question,
                        "retrieval_latency_sec": retrieval_latency,
                        "rag_latency_sec": None,
                        "corag_latency_sec": None,
                        "similarity_ratio": None,
                        "rag_answer": "",
                        "corag_answer": "",
                        "context_count": len(context_chunks),
                        "source_summary": [],
                        "error": f"RAG generation failed: {exc}",
                    }
                )
                continue

            try:
                corag_start = time.perf_counter()
                corag_answer, corag_context, corag_trace = generate_corag_answer(
                    question=question,
                    base_chunks=context_chunks,
                    base_answer=rag_answer,
                    rounds=corag_rounds,
                    memory_turns=[],
                    original_question=question,
                    use_hybrid_search=use_hybrid_search,
                    metadata_filters=document_filter,
                )
                corag_latency = round(time.perf_counter() - corag_start, 4)
            except Exception as exc:
                per_question_results.append(
                    {
                        "question": question,
                        "retrieval_latency_sec": retrieval_latency,
                        "rag_latency_sec": rag_latency,
                        "corag_latency_sec": None,
                        "similarity_ratio": None,
                        "rag_answer": rag_answer,
                        "corag_answer": "",
                        "context_count": len(context_chunks),
                        "source_summary": [],
                        "error": f"Co-RAG generation failed: {exc}",
                    }
                )
                continue

            similarity_ratio = round(SequenceMatcher(None, rag_answer, corag_answer).ratio(), 4)
            rag_score = round(sum(float(c.get("score", 0.0)) for c in context_chunks) / max(len(context_chunks), 1), 4)
            corag_score = round(sum(float(c.get("score", 0.0)) for c in corag_context) / max(len(corag_context), 1), 4)

            rag_latencies.append(rag_latency)
            corag_latencies.append(corag_latency)
            similarities.append(similarity_ratio)
            rag_scores.append(rag_score)
            corag_scores.append(corag_score)

            per_question_results.append(
                {
                    "question": question,
                    "retrieval_latency_sec": retrieval_latency,
                    "rag_latency_sec": rag_latency,
                    "corag_latency_sec": corag_latency,
                    "similarity_ratio": similarity_ratio,
                    "rag_answer": rag_answer,
                    "corag_answer": corag_answer,
                    "context_count": len(context_chunks),
                    "source_summary": _build_document_source_summary(corag_context),
                    "corag_trace": corag_trace,
                }
            )

        report_rows.append(
            {
                "document_id": document.get("document_id"),
                "source": source,
                "file_type": document.get("file_type"),
                "upload_date": document.get("upload_date"),
                "questions_count": len(questions),
                "avg_rag_latency_sec": round(sum(rag_latencies) / len(rag_latencies), 4) if rag_latencies else None,
                "avg_corag_latency_sec": round(sum(corag_latencies) / len(corag_latencies), 4) if corag_latencies else None,
                "avg_similarity_ratio": round(sum(similarities) / len(similarities), 4) if similarities else None,
                "avg_rag_retrieval_score": round(sum(rag_scores) / len(rag_scores), 4) if rag_scores else None,
                "avg_corag_retrieval_score": round(sum(corag_scores) / len(corag_scores), 4) if corag_scores else None,
                "question_results": per_question_results,
            }
        )

    report_rows.sort(key=lambda item: (item.get("avg_similarity_ratio") is None, item.get("avg_similarity_ratio", 1.0)))

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "corag_rounds": corag_rounds,
        "use_hybrid_search": config.USE_HYBRID_SEARCH if use_hybrid_search is None else use_hybrid_search,
        "base_metadata_filters": base_filters,
        "questions": questions,
        "rows": report_rows,
        "summary": {
            "documents_compared": len(report_rows),
            "questions_per_document": len(questions),
            "avg_similarity_ratio": round(
                sum(r["avg_similarity_ratio"] for r in report_rows if r.get("avg_similarity_ratio") is not None)
                / max(len([r for r in report_rows if r.get("avg_similarity_ratio") is not None]), 1),
                4,
            ) if report_rows else None,
        },
    }

    try:
        report_path = _persist_report(report_payload)
    except Exception as exc:
        return error_response("Failed to persist report", 500, details={"detail": str(exc)})

    return success_response(
        data={"report": report_payload, "report_path": report_path},
        message="RAG vs Co-RAG document report exported successfully",
    )