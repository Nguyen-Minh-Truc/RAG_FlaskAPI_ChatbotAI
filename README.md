# RAG Learning Starter (Skeleton Only)

This project is a minimal starter structure to learn Retrieval-Augmented Generation (RAG) step by step.

Important:

- This repository intentionally does not implement full RAG logic.
- Each module contains TODO placeholders so you can implement and understand each stage yourself.

## Tech Stack

- Python 3.10+
- Flask (REST API)
- LangChain
- Ollama + Qwen (local, no API key)

## Project Structure

- main.py: Flask entry point
- app/api/routes.py: REST endpoints GET /health, POST /preprocess/pdf, POST /upload, POST /ingest/pdf, and POST /ask
- app/rag/loader.py: load uploaded PDF from request (TODO)
- app/rag/chunker.py: split text into chunks (TODO)
- app/rag/embeddings.py: create embeddings (TODO)
- app/rag/vectorstore.py: FAISS setup/search (TODO)
- app/rag/retriever.py: retrieval logic (TODO + simple logging placeholder)
- app/llm/llm_service.py: LLM abstraction layer (TODO)
- app/config.py: configuration and .env loading
- data/sample.txt: sample source document

## RAG Flow

User question -> embedding -> similarity search -> retrieve top-k chunks -> send to LLM -> return answer

This flow is documented in comments across modules so you can implement each stage in isolation.

## API Contract

### GET /health

Response body:
{
"status": "ok"
}

### POST /preprocess/pdf

Request type:
multipart/form-data

Form fields:

- file: PDF file

Response body (skeleton):
{
"message": "string",
"preprocessing": [],
"next_step": "string"
}

### POST /upload

Alias of POST /preprocess/pdf for convenience.

### POST /ingest/pdf

Request type:
multipart/form-data

Form fields:

- file: PDF file

Response body (example):
{
"message": "PDF ingestion completed successfully.",
"documents_loaded": 1,
"chunks_created": 10,
"embeddings_created": 10,
"vectorstore": {}
}

### POST /ask

Request body:
{
"question": "string"
}

Response body:
{
"answer": "string",
"context": []
}

## Quick Start

1. Create and activate virtual environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Pull local model with Ollama:
   ollama pull qwen2.5:7b
4. Set environment values in .env.
5. Run app:
   python main.py

## Local LLM Note

- This skeleton is configured for local Qwen by default via Ollama.
- You can keep OPENAI_API_KEY empty when running local only.

## Suggested Learning Order

1. Implement loader.py to parse uploaded PDF into text documents.
2. Implement chunker.py and test chunk quality.
3. Implement embeddings.py for chunk and question embeddings.
4. Implement vectorstore.py with FAISS indexing and similarity search.
5. Wire ingestion flow in POST /upload (loader -> chunker -> embeddings -> vectorstore).
6. Implement retriever.py to return scored top-k chunks.
7. Implement llm_service.py prompt + model call.
8. Wire ask-time orchestration in POST /ask.
