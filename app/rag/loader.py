"""Document loading stage for RAG (skeleton only)."""

from pypdf import PdfReader
from werkzeug.datastructures import FileStorage


def load_uploaded_pdf(uploaded_file: FileStorage) -> list[dict]:
    """
    Primary ingestion path for this project: load PDF from user upload request.

    This function keeps the implementation intentionally small and focused:
    - Read the uploaded PDF stream
    - Extract text page by page
    - Return a list of document-like dictionaries for the next RAG stages
    """
    if uploaded_file is None:
        raise ValueError("uploaded_file is required")

    # Reset the stream so the PDF reader starts from the beginning every time.
    uploaded_file.stream.seek(0)

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
                    "source": uploaded_file.filename,
                    "page": page_number,
                    "content_type": uploaded_file.content_type,
                },
            }
        )

    return documents


# def load_local_documents() -> list[dict]:
#     """
#     Optional learning path: load local files from data/ for easier debugging.

#     Keep this as secondary path. The main goal remains PDF upload from API.
#     """
#     # TODO: Implement local file discovery and parsing if needed.
#     return []
