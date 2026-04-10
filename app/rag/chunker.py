"""Chunking stage for RAG (skeleton only)."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_into_chunks(documents: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """
    Split loaded documents into smaller retrieval-friendly chunks.

    Input shape from loader:
    - [{"text": "...", "metadata": {...}}]

    Output shape for downstream steps:
    - [{"text": "chunk text", "metadata": {..., "chunk_index": 0}}]
    """
    if chunk_size <= 0:
        chunk_size = 1000  # Default chunk size if invalid input is provided

    if chunk_overlap < 0:
        chunk_overlap = 200  # Default chunk overlap if invalid input is provided

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", " ", ""],
    )

    chunked_documents: list[dict] = []

    for document in documents:
        text = document.get("text", "")
        metadata = document.get("metadata", {}).copy()

        if not text.strip():
            continue

        chunks = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(chunks):
            cleaned_chunk = chunk_text.strip()
            if not cleaned_chunk:
                continue

            chunked_documents.append(
                {
                    "text": cleaned_chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    },
                }
            )

    return chunked_documents
