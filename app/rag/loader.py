"""Document loading stage for RAG (skeleton only)."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from langchain_community.document_loaders import UnstructuredFileLoader
from pypdf import PdfReader
from werkzeug.datastructures import FileStorage


SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".doc", ".docx"}


def _normalize_document_metadata(uploaded_file: FileStorage, extension: str, metadata: dict[str, Any] | None) -> dict:
    """Build a normalized document-level metadata object."""
    base_metadata: dict[str, Any] = {
        "source": uploaded_file.filename,
        "file_type": extension.lstrip("."),
        "content_type": uploaded_file.content_type,
    }

    if metadata:
        base_metadata.update(metadata)

    return base_metadata


def _load_pdf(uploaded_file: FileStorage, document_metadata: dict[str, Any] | None = None) -> list[dict]:
    """Load PDF content page by page."""
    uploaded_file.stream.seek(0)
    extension = Path(uploaded_file.filename or "").suffix.lower()
    normalized_metadata = _normalize_document_metadata(uploaded_file, extension, document_metadata)

    reader = PdfReader(uploaded_file.stream)
    documents: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        cleaned_text = page_text.strip()

        if not cleaned_text:
            continue

        documents.append(
            {
                "text": cleaned_text,
                "metadata": {
                    **normalized_metadata,
                    "page": page_number,
                },
            }
        )

    return documents


def _load_word(
    uploaded_file: FileStorage,
    extension: str,
    document_metadata: dict[str, Any] | None = None,
) -> list[dict]:
    """Load DOC/DOCX content using Unstructured loader."""
    uploaded_file.stream.seek(0)
    file_bytes = uploaded_file.read()
    normalized_metadata = _normalize_document_metadata(uploaded_file, extension, document_metadata)

    with NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = UnstructuredFileLoader(temp_path)
        loaded_docs = loader.load()

        documents: list[dict] = []
        for idx, doc in enumerate(loaded_docs, start=1):
            text = (doc.page_content or "").strip()
            if not text:
                continue

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        **normalized_metadata,
                        "part": idx,
                        **(doc.metadata or {}),
                    },
                }
            )

        return documents
    finally:
        Path(temp_path).unlink(missing_ok=True)


def load_uploaded_document(
    uploaded_file: FileStorage,
    document_metadata: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Primary ingestion path for this project: load uploaded file from user request.

    This function keeps the implementation intentionally small and focused:
    - Support PDF, DOC and DOCX
    - Extract readable text
    - Return a list of document-like dictionaries for the next RAG stages
    """
    if uploaded_file is None:
        raise ValueError("uploaded_file is required")

    filename = (uploaded_file.filename or "").strip()
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValueError("Only .pdf, .doc and .docx files are supported")

    if extension == ".pdf":
        return _load_pdf(uploaded_file, document_metadata=document_metadata)

    return _load_word(uploaded_file, extension, document_metadata=document_metadata)


# def load_local_documents() -> list[dict]:
#     """
#     Optional learning path: load local files from data/ for easier debugging.

#     Keep this as secondary path. The main goal remains PDF upload from API.
#     """
#     # TODO: Implement local file discovery and parsing if needed.
#     return []
