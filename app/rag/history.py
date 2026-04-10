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


def create_conversation(upload_filename: str) -> str:
    """Create a new conversation session and persist metadata."""
    conversation_id = _next_conversation_id()
    now = _utc_now_iso()
    payload = {
        "conversation_id": conversation_id,
        "upload_filename": upload_filename,
        "created_at": now,
        "updated_at": now,
        "turn_count": 0,
        "turns": [],
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
    context: list[dict],
) -> dict:
    """Append one Q&A turn into an existing conversation."""
    history = load_conversation_history(conversation_id)

    turn_id = history["turn_count"] + 1
    turn = {
        "turn_id": turn_id,
        "timestamp": _utc_now_iso(),
        "question": question,
        "answer": answer,
        "context": context,
    }

    history["turns"].append(turn)
    history["turn_count"] = turn_id
    history["updated_at"] = _utc_now_iso()

    _conversation_path(conversation_id).write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return turn


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
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
                "turn_count": payload.get("turn_count", 0),
            }
        )

    summaries.sort(key=lambda item: int(str(item.get("conversation_id", "0"))), reverse=True)
    return summaries