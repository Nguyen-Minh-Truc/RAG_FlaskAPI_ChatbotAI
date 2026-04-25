"""Conversation history persistence for per-upload chat sessions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app import config


def _history_dir() -> Path:
    history_dir = Path(config.CHAT_HISTORY_DIR)
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


def _conversation_path(conversation_id: str) -> Path:
    return _history_dir() / f"{conversation_id}.json"


def _next_conversation_id() -> str:
    """Generate the next numeric conversation id as a string."""
    max_id = 0
    for path in _history_dir().glob("*.json"):
        stem = path.stem
        if stem.isdigit():
            max_id = max(max_id, int(stem))
    return str(max_id + 1)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_memory_turn(turn: dict) -> dict:
    return {
        "turn_id": turn.get("turn_id"),
        "timestamp": turn.get("timestamp"),
        "question": turn.get("question", ""),
        "answer": turn.get("answer", ""),
        "corag_answer": turn.get("corag_answer", ""),
    }


def _recent_turns(turns: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []

    return [_build_memory_turn(turn) for turn in turns[-limit:]]


def create_conversation(upload_filename: str) -> str:
    """Create a new conversation session and persist metadata."""
    conversation_id = _next_conversation_id()
    now = _utc_now_iso()
    payload = {
        "conversation_id": conversation_id,
        "upload_filename": upload_filename,
        "uploaded_documents": [],
        "created_at": now,
        "updated_at": now,
        "turn_count": 0,
        "turns": [],
        "recent_turns": [],
    }
    _conversation_path(conversation_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return conversation_id


def load_conversation_history(conversation_id: str) -> dict:
    """Load conversation history by id."""
    path = _conversation_path(conversation_id)
    if not path.exists():
        raise FileNotFoundError(f"Conversation not found: {conversation_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def append_conversation_turn(
    conversation_id: str,
    question: str,
    answer: str,
    corag_answer: str | None,
    context: list[dict],
    memory_context: list[dict] | None = None,
    resolved_question: str | None = None,
    mode: str = "Compare",
    metadata_filters: dict | None = None,
    source_summary: list[dict] | None = None,
) -> dict:
    """Append one Q&A turn into an existing conversation."""
    history = load_conversation_history(conversation_id)

    turn_id = history["turn_count"] + 1
    turn = {
        "turn_id": turn_id,
        "timestamp": _utc_now_iso(),
        "question": question,
        "resolved_question": resolved_question or question,
        "answer": answer,
        "corag_answer": corag_answer,
        "mode": mode,
        "context": context,
        "metadata_filters": metadata_filters or {},
        "source_summary": source_summary or [],
        "memory_context": memory_context or [],
        "follow_up": bool(memory_context),
    }

    history["turns"].append(turn)
    history["turn_count"] = turn_id
    history["updated_at"] = _utc_now_iso()
    history["recent_turns"] = _recent_turns(history["turns"], limit=4)

    _conversation_path(conversation_id).write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return turn


def get_recent_turns(conversation_id: str, limit: int = 4) -> list[dict]:
    """Return the most recent turns for conversation memory."""
    history = load_conversation_history(conversation_id)
    return _recent_turns(history.get("turns", []), limit=limit)


def append_uploaded_documents(conversation_id: str, uploaded_documents: list[dict]) -> list[dict]:
    """Append uploaded document metadata into one conversation."""
    if not uploaded_documents:
        return []

    history = load_conversation_history(conversation_id)
    existing_docs = list(history.get("uploaded_documents", []))
    seen_ids = {str(doc.get("document_id", "")) for doc in existing_docs}

    for doc in uploaded_documents:
        document_id = str(doc.get("document_id", ""))
        if document_id and document_id in seen_ids:
            continue

        existing_docs.append(doc)
        if document_id:
            seen_ids.add(document_id)

    history["uploaded_documents"] = existing_docs
    history["updated_at"] = _utc_now_iso()

    _conversation_path(conversation_id).write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return existing_docs


def list_conversations() -> list[dict]:
    """Return conversation summaries sorted by newest conversation id first."""
    summaries: list[dict] = []

    for path in _history_dir().glob("*.json"):
        if not path.stem.isdigit():
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(
            {
                "conversation_id": payload.get("conversation_id", path.stem),
                "upload_filename": payload.get("upload_filename", ""),
                "filename": payload.get("upload_filename", ""),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
                "turn_count": payload.get("turn_count", 0),
            }
        )

    summaries.sort(key=lambda item: int(str(item.get("conversation_id", "0"))), reverse=True)
    return summaries


def delete_conversation(conversation_id: str) -> bool:
    """Delete one conversation by id. Return True if deleted, False if not found."""
    path = _conversation_path(conversation_id)
    if not path.exists():
        return False

    path.unlink()
    return True


def delete_all_conversations() -> int:
    """Delete all conversation files and return deleted file count."""
    deleted_count = 0
    for path in _history_dir().glob("*.json"):
        if path.is_file():
            path.unlink()
            deleted_count += 1

    return deleted_count