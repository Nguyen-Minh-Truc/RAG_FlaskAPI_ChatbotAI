"""Application configuration with .env support."""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


# Flask settings
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

# LLM / API settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:5b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# RAG tuning settings
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Persistence settings
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", str(BASE_DIR / "storage" / "vectorstore"))
CHAT_HISTORY_DIR = os.getenv("CHAT_HISTORY_DIR", str(BASE_DIR / "storage" / "conversations"))

# TODO: Add vector store persistence and index path settings.
