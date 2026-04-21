"""Hybrid retrieval utilities combining FAISS semantic search and BM25 keyword search."""

from __future__ import annotations

import hashlib
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from pydantic import ConfigDict

from app import config
from app.rag.embeddings import create_embeddings
from app.rag.embeddings import create_query_embedding
from app.rag.vectorstore import load_vectorstore, search_similar


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _normalize_text(text: str) -> str:
    """Normalize whitespace and strip surrounding spaces from text."""
    return " ".join((text or "").split()).strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 while tolerating punctuation and special characters."""
    normalized = _normalize_text(text).lower()
    tokens = _TOKEN_PATTERN.findall(normalized)
    return tokens or ([normalized] if normalized else [])


def _coerce_metadata(value: Any) -> dict:
    """Convert metadata-like values into a plain dictionary."""
    if isinstance(value, dict):
        return dict(value)

    return {}


def _coerce_document(item: Any) -> Document:
    """Convert a persisted chunk or LangChain document into a Document."""
    if isinstance(item, Document):
        return Document(page_content=item.page_content or "", metadata=_coerce_metadata(item.metadata))

    if isinstance(item, dict):
        page_content = item.get("text") or item.get("page_content") or ""
        metadata = _coerce_metadata(item.get("metadata"))
        return Document(page_content=str(page_content), metadata=metadata)

    return Document(page_content=str(item or ""), metadata={})


def _serialize_documents(documents: list[Document]) -> str:
    """Serialize documents into a stable cache key payload."""
    payload = [
        {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _documents_signature(documents: list[Document]) -> str:
    """Build a stable signature for a list of documents."""
    serialized = _serialize_documents(documents)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _store_fingerprint(persist_dir: str | Path | None) -> str:
    """Fingerprint the persisted vector store for cache invalidation."""
    store_dir = Path(persist_dir or config.VECTORSTORE_DIR)
    documents_path = store_dir / "documents.json"
    index_path = store_dir / "faiss.index"

    if not documents_path.exists():
        return f"missing-docs:{store_dir}"

    documents_stat = documents_path.stat()
    index_part = "missing-index"
    if index_path.exists():
        index_stat = index_path.stat()
        index_part = f"{index_stat.st_size}:{index_stat.st_mtime_ns}"

    return f"{documents_path}:{documents_stat.st_size}:{documents_stat.st_mtime_ns}:{index_part}"


def _precision_recall_mrr(retrieved: list[Document], ground_truth_docs: list[Any], k: int) -> dict:
    """Compute retrieval metrics against a ground-truth document set."""
    truth_identities = {_document_identity(_coerce_document(item)) for item in ground_truth_docs if _normalize_text(_coerce_document(item).page_content)}

    if not truth_identities:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

    hits = 0
    first_relevant_rank = 0

    for rank, document in enumerate(retrieved[:k], start=1):
        identity = _document_identity(document)
        if identity in truth_identities:
            hits += 1
            if first_relevant_rank == 0:
                first_relevant_rank = rank

    precision = hits / float(max(k, 1))
    recall = hits / float(len(truth_identities))
    mrr = 1.0 / float(first_relevant_rank) if first_relevant_rank else 0.0

    return {
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "mrr": round(mrr, 4),
    }


def _document_identity(document: Document) -> str:
    """Build a matching key for a document using metadata when available."""
    metadata = document.metadata or {}
    source = metadata.get("source")
    page = metadata.get("page")
    chunk_index = metadata.get("chunk_index")

    if source is not None or page is not None or chunk_index is not None:
        return f"source={source}|page={page}|chunk_index={chunk_index}"

    return _normalize_text(document.page_content).lower()


def _print_comparison_table(result: dict) -> None:
    """Print a compact comparison table for retrieval metrics."""
    headers = ["mode", "precision@k", "recall@k", "mrr", "latency_ms", "results"]
    rows = [result["vector"], result["bm25"], result["ensemble"]]
    widths = [len(header) for header in headers]

    for row in rows:
        values = [
            row["mode"],
            f"{row['precision_at_k']:.4f}",
            f"{row['recall_at_k']:.4f}",
            f"{row['mrr']:.4f}",
            f"{row['latency_ms']:.2f}",
            str(len(row.get("documents", []))),
        ]
        widths = [max(width, len(value)) for width, value in zip(widths, values)]

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(separator)
    print("| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths)) + " |")
    print(separator)

    for row in rows:
        values = [
            row["mode"],
            f"{row['precision_at_k']:.4f}",
            f"{row['recall_at_k']:.4f}",
            f"{row['mrr']:.4f}",
            f"{row['latency_ms']:.2f}",
            str(len(row.get("documents", []))),
        ]
        print("| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |")

    print(separator)


class _FAISSRetriever(BaseRetriever):
    """Lightweight retriever that wraps the existing FAISS similarity search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any | None = None
    documents: list[dict] = []
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: Any | None = None) -> list[Document]:
        """Return the top-k FAISS matches as LangChain documents."""
        if not query or not query.strip():
            return []

        if self.index is None or not self.documents:
            return []

        query_vector = create_query_embedding(query)
        results = search_similar(
            index=self.index,
            query_vector=query_vector,
            top_k=self.k,
            documents=self.documents,
        )

        return [
            Document(
                page_content=item.get("text", ""),
                metadata={**_coerce_metadata(item.get("metadata")), "score": float(item.get("score", 0.0)), "retriever": "faiss"},
            )
            for item in results
            if _normalize_text(item.get("text", ""))
        ]


class _EmptyRetriever(BaseRetriever):
    """Retriever that always returns an empty result set."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: int = 0

    def _get_relevant_documents(self, query: str, *, run_manager: Any | None = None) -> list[Document]:
        """Return no documents for any query."""
        return []


@lru_cache(maxsize=8)
def _build_cached_bundle(
    cache_key: str,
    persist_dir: str,
    documents_payload: str | None,
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
) -> dict:
    """Build and cache the vector retriever, BM25 retriever, and ensemble retriever."""
    if documents_payload is None:
        index, raw_documents = load_vectorstore(persist_dir=persist_dir)
        documents = [_coerce_document(item) for item in raw_documents]
        raw_documents_payload = raw_documents
    else:
        raw_documents_payload = json.loads(documents_payload)
        documents = [_coerce_document(item) for item in raw_documents_payload]
        filtered_documents = [document for document in documents if _normalize_text(document.page_content)]
        if filtered_documents:
            vectors = create_embeddings([document.page_content for document in filtered_documents])
            matrix = np.asarray(vectors, dtype=np.float32)
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            raw_documents_payload = [
                {"text": document.page_content, "metadata": document.metadata}
                for document in filtered_documents
            ]
        else:
            index = None
            raw_documents_payload = []

    filtered_documents = [document for document in documents if _normalize_text(document.page_content)]

    vector_retriever = (
        _FAISSRetriever(index=index, documents=raw_documents_payload, k=top_k)
        if index is not None and raw_documents_payload
        else _EmptyRetriever()
    )

    if filtered_documents:
        bm25_retriever = BM25Retriever.from_documents(
            filtered_documents,
            preprocess_func=_tokenize,
        )
        bm25_retriever.k = top_k
    else:
        bm25_retriever = _EmptyRetriever()

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )

    return {
        "documents": documents,
        "vector_retriever": vector_retriever,
        "bm25_retriever": bm25_retriever,
        "ensemble_retriever": ensemble_retriever,
    }


class HybridSearchRetriever:
    """Hybrid retriever combining FAISS semantic search and BM25 keyword search."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        top_k: int = 5,
        weights: tuple[float, float] = (0.5, 0.5),
        documents: list[Any] | None = None,
    ) -> None:
        """Initialize the hybrid retriever configuration."""
        self.persist_dir = str(Path(persist_dir or config.VECTORSTORE_DIR))
        self.top_k = top_k
        self.weights = weights
        self._source_documents = documents
        self._bundle: dict[str, Any] | None = None

    def build(self, documents: list[Any] | None = None) -> "HybridSearchRetriever":
        """Build or reuse the cached FAISS, BM25, and ensemble retrievers."""
        source_documents = documents if documents is not None else self._source_documents

        if source_documents is None:
            cache_key = _store_fingerprint(self.persist_dir)
            bundle = _build_cached_bundle(
                cache_key=cache_key,
                persist_dir=self.persist_dir,
                documents_payload=None,
                top_k=self.top_k,
                vector_weight=self.weights[0],
                bm25_weight=self.weights[1],
            )
        else:
            coerced_documents = [_coerce_document(item) for item in source_documents]
            serialized_documents = _serialize_documents(coerced_documents)
            cache_key = _documents_signature(coerced_documents)
            bundle = _build_cached_bundle(
                cache_key=cache_key,
                persist_dir=self.persist_dir,
                documents_payload=serialized_documents,
                top_k=self.top_k,
                vector_weight=self.weights[0],
                bm25_weight=self.weights[1],
            )

        self._bundle = bundle
        return self

    def _ensure_bundle(self) -> dict[str, Any]:
        """Return the cached retriever bundle, building it on demand."""
        if self._bundle is None:
            self.build()

        if self._bundle is None:
            raise RuntimeError("Hybrid retriever bundle could not be built")

        return self._bundle

    def _invoke_retriever(self, retriever: Any, query: str) -> list[Document]:
        """Invoke a LangChain retriever and normalize the result to documents."""
        if not query or not query.strip():
            return []

        if hasattr(retriever, "invoke"):
            documents = retriever.invoke(query)
        else:
            documents = retriever.get_relevant_documents(query)

        return [_coerce_document(document) for document in documents if _normalize_text(_coerce_document(document).page_content)]

    def _document_dicts_from_documents(self, documents: list[Document], mode: str) -> list[dict]:
        """Convert LangChain documents into the stable response shape."""
        normalized: list[dict] = []
        for rank, document in enumerate(documents[: self.top_k], start=1):
            metadata = _coerce_metadata(document.metadata)
            metadata.setdefault("retriever", mode)
            normalized.append(
                {
                    "text": document.page_content,
                    "score": float(metadata.get("score", 1.0 / rank)),
                    "metadata": metadata,
                }
            )

        return normalized

    def retrieve(
        self,
        query: str | None = None,
        k: int | None = None,
        question: str | None = None,
    ) -> list[dict]:
        """Retrieve documents using the ensemble retriever and return stable chunk dicts."""
        effective_query = query if query is not None else question
        if not effective_query or not effective_query.strip():
            return []

        bundle = self._ensure_bundle()
        effective_k = max(int(k or self.top_k), 1)
        self.top_k = effective_k

        vector_retriever = bundle["vector_retriever"]
        bm25_retriever = bundle["bm25_retriever"]
        ensemble_retriever = bundle["ensemble_retriever"]

        vector_retriever.k = effective_k
        bm25_retriever.k = effective_k

        documents = self._invoke_retriever(ensemble_retriever, effective_query)
        if not documents:
            return []

        return self._document_dicts_from_documents(documents, mode="ensemble")

    def _mode_results(self, mode: str, query: str, k: int) -> list[Document]:
        """Run one retrieval mode and return the raw LangChain documents."""
        bundle = self._ensure_bundle()
        vector_retriever = bundle["vector_retriever"]
        bm25_retriever = bundle["bm25_retriever"]
        ensemble_retriever = bundle["ensemble_retriever"]

        vector_retriever.k = k
        bm25_retriever.k = k

        if mode == "vector":
            return self._invoke_retriever(vector_retriever, query)

        if mode == "bm25":
            return self._invoke_retriever(bm25_retriever, query)

        if mode == "ensemble":
            return self._invoke_retriever(ensemble_retriever, query)

        raise ValueError(f"Unsupported retrieval mode: {mode}")

    def compare_performance(
        self,
        query: str,
        ground_truth_docs: list[Any] | None = None,
        k: int = 5,
    ) -> dict:
        """Compare vector, BM25, and ensemble retrieval on the same query.

        Returns retrieval metrics and latency for each retrieval mode.
        """
        if not query or not query.strip():
            return {"query": query, "k": k, "vector": {}, "bm25": {}, "ensemble": {}}

        if ground_truth_docs is None:
            ground_truth_docs = []

        comparison: dict[str, Any] = {"query": query, "k": k}

        for mode in ("vector", "bm25", "ensemble"):
            started_at = time.perf_counter()
            documents = self._mode_results(mode=mode, query=query, k=k)
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            metrics = _precision_recall_mrr(documents, ground_truth_docs, k=k)
            comparison[mode] = {
                "mode": mode,
                "documents": [
                    {
                        "text": document.page_content,
                        "metadata": _coerce_metadata(document.metadata),
                    }
                    for document in documents[:k]
                ],
                "latency_ms": round(latency_ms, 2),
                **metrics,
            }

        _print_comparison_table(comparison)
        return comparison


def _demo_documents() -> list[dict]:
    """Return sample documents for the module demo."""
    return [
        {
            "text": "Hybrid search combines semantic vector matching with keyword BM25 retrieval.",
            "metadata": {"source": "demo-1", "page": 1, "chunk_index": 0},
        },
        {
            "text": "FAISS is efficient for dense vector similarity search over embeddings.",
            "metadata": {"source": "demo-2", "page": 1, "chunk_index": 1},
        },
        {
            "text": "BM25 works well for exact keyword matching, special terms, and short queries.",
            "metadata": {"source": "demo-3", "page": 2, "chunk_index": 0},
        },
    ]


if __name__ == "__main__":
    """Run a small demo of hybrid search on in-memory sample documents."""
    retriever = HybridSearchRetriever(top_k=2, documents=_demo_documents())
    retriever.build()

    query = "How does hybrid search handle BM25 and vector matching?"
    print("\nRetrieved documents:\n")
    for idx, item in enumerate(retriever.retrieve(query), start=1):
        print(f"{idx}. score={item['score']:.4f} | source={item['metadata'].get('source')} | text={item['text']}")

    print("\nComparison:\n")
    retriever.compare_performance(
        query=query,
        ground_truth_docs=[_demo_documents()[0], _demo_documents()[1]],
        k=2,
    )