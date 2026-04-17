"""REST API routes for RAG learning skeleton."""

from pathlib import Path

from flask import Blueprint, request

from app import config
from app.corag.pipeline import generate_corag_answer
from app.rag.chunker import split_into_chunks
from app.rag.embeddings import create_embeddings
from app.rag.history import (
    append_conversation_turn,
    create_conversation,
    delete_all_conversations,
    delete_conversation,
    list_conversations,
    load_conversation_history,
)
from app.rag.loader import SUPPORTED_UPLOAD_EXTENSIONS, load_uploaded_document
from app.rag.retriever import retrieve_top_k_chunks
from app.rag.vectorstore import save_embeddings_and_vectorstore
from app.llm.llm_service import generate_answer
from app.api.response import error_response, success_response

api_bp = Blueprint("api", __name__)

MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 5000
MIN_CHUNK_OVERLAP = 0
MAX_CHUNK_OVERLAP = 2000

@api_bp.get("/api/health")
def health_check():
    """Simple health endpoint for quick service checks."""
    return success_response(data={"status": "ok"}, message="Service is healthy")


@api_bp.get("/api/conversations")
def get_conversations():
    """List all conversation summaries."""
    conversations = list_conversations()
    return success_response(
        data=conversations,
        message="Conversation list fetched successfully",
    )


@api_bp.get("/api/conversations/<conversation_id>")
def get_conversation_detail(conversation_id: str):
    """Get full details of one conversation by id."""
    try:
        history = load_conversation_history(conversation_id=conversation_id)
    except FileNotFoundError:
        return error_response("Conversation not found.", 404)
    except Exception as exc:
        return error_response("Failed to load conversation", 500, details={"detail": str(exc)})

    turns = history.get("turns", [])
    sanitized_turns = []
    for turn in turns:
        sanitized_turn = dict(turn)
        sanitized_turn.pop("context", None)
        sanitized_turns.append(sanitized_turn)

    history["turns"] = sanitized_turns

    return success_response(data=history, message="Conversation detail fetched successfully")


@api_bp.delete("/api/conversations/<conversation_id>")
def remove_conversation(conversation_id: str):
    """Delete one conversation history by id."""
    try:
        deleted = delete_conversation(conversation_id)
    except Exception as exc:
        return error_response("Failed to delete conversation", 500, details={"detail": str(exc)})

    if not deleted:
        return error_response("Conversation not found.", 404)

    return success_response(
        data={"conversation_id": conversation_id},
        message="Conversation deleted successfully",
    )


@api_bp.delete("/api/conversations")
def remove_all_conversations():
    """Delete all conversation histories."""
    try:
        deleted_count = delete_all_conversations()
    except Exception as exc:
        return error_response("Failed to delete all conversations", 500, details={"detail": str(exc)})

    return success_response(
        data={"deleted": deleted_count},
        message="All conversations deleted successfully",
    )


def _validate_upload_file():
    """Validate multipart upload and return a supported file object."""
    uploaded_file = request.files.get("file")

    if uploaded_file is None:
        return None, error_response("Field 'file' is required.", 400)

    if not uploaded_file.filename:
        return None, error_response("Filename is required.", 400)

    extension = Path(uploaded_file.filename).suffix.lower()
    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        return None, error_response(f"Unsupported file type. Allowed: {allowed}", 400)

    return uploaded_file, None


def _parse_upload_chunk_params() -> tuple[int, int] | tuple[None, tuple]:
    """Parse optional chunk params from multipart form with validation."""
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
            f"Field 'chunk_size' must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}.",
            400,
        )

    if not (MIN_CHUNK_OVERLAP <= chunk_overlap <= MAX_CHUNK_OVERLAP):
        return None, error_response(
            f"Field 'chunk_overlap' must be between {MIN_CHUNK_OVERLAP} and {MAX_CHUNK_OVERLAP}.",
            400,
        )

    if chunk_overlap >= chunk_size:
        return None, error_response("Field 'chunk_overlap' must be smaller than 'chunk_size'.", 400)

    return (chunk_size, chunk_overlap), None


@api_bp.post("/api/upload")
def ingest_pdf():
    """
    POST /upload
    Content-Type: multipart/form-data
    Field: file (PDF/DOC/DOCX)

    Ingestion flow:
    uploaded file -> load text -> chunk -> embed -> store in FAISS
    """
    uploaded_file, validation_error = _validate_upload_file()
    if validation_error:
        return validation_error

    chunk_params, chunk_error = _parse_upload_chunk_params()
    if chunk_error:
        return chunk_error

    chunk_size, chunk_overlap = chunk_params

    conversation_id = create_conversation(uploaded_file.filename or "uploaded.pdf")

    try:
        documents = load_uploaded_document(uploaded_file)
    except ValueError as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        extension = Path(uploaded_file.filename or "").suffix.lower() or "unknown"
        return error_response(
            "Failed to parse uploaded file",
            500,
            details={
                "detail": str(exc),
                "filename": uploaded_file.filename,
                "extension": extension,
                "hint": "With .doc/.docx files, make sure parser dependencies are installed; with scanned PDFs, OCR may be required.",
            },
        )

    if not documents:
        return error_response(
            "Failed to parse uploaded file",
            400,
            details={
                "detail": "No readable text was extracted from the uploaded file.",
                "filename": uploaded_file.filename,
                "hint": "Try another file, export as text-based PDF/DOCX, or run OCR first if the file is image-only.",
            },
        )
    chunks = split_into_chunks(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunk_texts = [chunk.get("text", "") for chunk in chunks]
    vectors = create_embeddings(chunk_texts)

    store_info = save_embeddings_and_vectorstore(
        documents=chunks,
        vectors=vectors,
        persist_dir=config.VECTORSTORE_DIR,
    )

    return success_response(
        data={
            "conversation_id": conversation_id,
            "documents": len(documents),
            "chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        message="PDF ingestion completed successfully",
    )


@api_bp.post("/api/ask")
def ask_question():
    """
    POST /ask
    Request: {"question": "string"}
    Response: {"answer": "string", "context": []}

    Learning flow (skeleton only):
    User question -> embedding -> similarity search -> retrieve top-k chunks -> send to LLM -> return answer
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error_response("Invalid JSON body.", 400)

    question = payload.get("question")
    conversation_id = payload.get("conversation_id")
    corag_rounds = payload.get("corag_rounds", 1)

    if not isinstance(question, str) or not question.strip():
        return error_response("Field 'question' is required and must be a non-empty string.", 400)

    if not isinstance(corag_rounds, int):
        return error_response("Field 'corag_rounds' must be an integer.", 400)

    question = question.strip()

    # conversation_id is optional: if missing, use latest conversation.
    if isinstance(conversation_id, str) and conversation_id.strip():
        conversation_id = conversation_id.strip()
    else:
        conversations = list_conversations()
        if not conversations:
            return error_response(
                "No conversation found. Please upload a file first via POST /api/upload.",
                400,
            )
        conversation_id = str(conversations[0].get("conversation_id"))

    # RAG step 1-3:
    # User question -> embedding -> similarity search -> retrieve top-k chunks
    try:
        context_chunks = retrieve_top_k_chunks(question=question, top_k=config.TOP_K)
    except FileNotFoundError:
        return error_response(
            "Vector store not found. Please upload a file first via POST /api/upload.",
            400,
        )
    except Exception as exc:
        return error_response("Retrieval failed", 500, details={"detail": str(exc)})

    # RAG step 4-5:
    # send top-k context + question to LLM -> return generated answer
    try:
        rag_answer = generate_answer(question=question, context_chunks=context_chunks)
    except Exception as exc:
        return error_response("RAG generation failed", 500, details={"detail": str(exc)})

    try:
        corag_answer, _, corag_trace = generate_corag_answer(
            question=question,
            base_chunks=context_chunks,
            base_answer=rag_answer,
            rounds=corag_rounds,
        )
    except Exception as exc:
        return error_response("Co-RAG generation failed", 500, details={"detail": str(exc)})

    try:
        turn = append_conversation_turn(
            conversation_id=conversation_id,
            question=question,
            answer=rag_answer,
            corag_answer=corag_answer,
            context=context_chunks,
        )
    except FileNotFoundError:
        return error_response(
            "Conversation not found. Please upload a file first to create a conversation.",
            400,
        )
    except Exception as exc:
        return error_response("Failed to persist conversation history", 500, details={"detail": str(exc)})

    return success_response(
        data={
            "conversation_id": conversation_id,
            "turn_id": turn.get("turn_id"),
            "rag_answer": rag_answer,
            "corag_answer": corag_answer,
            "corag_trace": corag_trace,
        },
        message="RAG and Co-RAG answers generated successfully",
    )
