"""ChromaDB vector store for document retrieval.

Wraps ChromaDB's persistent client with:
  - Automatic embedding via sentence-transformers (local, no API key)
  - Add chunks from the ingestion pipeline
  - Semantic search with top-k retrieval
  - Delete by document ID or source filename
  - Collection stats and health check

The embedding model (default: all-MiniLM-L6-v2) runs locally on CPU.
It's 80MB, loads in ~2s, and embeds at ~500 chunks/sec on a modern laptop.
No GPU needed, no API key needed, no network calls during inference.

Usage:
    store = VectorStore(config)
    store.add_chunks(chunks, document_id="doc_001")
    results = store.search("how does backpropagation work?", top_k=5)
    store.delete_document("doc_001")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from src.rag.chunker import Chunk
from src.monitoring.logging import logger


@dataclass
class SearchResult:
    text: str
    score: float  # cosine similarity (higher = more relevant)
    chunk_index: int
    document_id: str
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """ChromaDB-backed vector store with sentence-transformer embeddings."""

    def __init__(self, config: dict | None = None):
        rag_cfg = (config or {}).get("rag", {})
        self._collection_name = rag_cfg.get("collection_name", "documents")
        self._top_k = rag_cfg.get("top_k", 5)
        self._persist_dir = rag_cfg.get("persist_directory", "data/chromadb")
        self._model_name = rag_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

        # Lazy-load heavy dependencies so import doesn't block startup
        self._client = None
        self._collection = None
        self._embed_fn = None

    def _ensure_initialized(self) -> None:
        """Lazy init — first call loads the embedding model + creates collection."""
        if self._collection is not None:
            return

        import chromadb

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # Build the embedding function
        self._embed_fn = self._build_embedding_function()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        count = self._collection.count()
        logger.info(
            f"VectorStore initialized: collection='{self._collection_name}', "
            f"model='{self._model_name}', existing_chunks={count}"
        )

    def _build_embedding_function(self):
        """Create a ChromaDB-compatible embedding function from sentence-transformers."""
        from chromadb.utils import embedding_functions

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self._model_name,
        )

    # ── Add ────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], document_id: str) -> int:
        """Add chunks to the vector store. Returns number of chunks added.

        Each chunk gets a unique ID: "{document_id}_chunk_{index}"
        Metadata includes source filename, chunk index, and token count.
        """
        self._ensure_initialized()

        if not chunks:
            return 0

        start = time.monotonic()
        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = f"{document_id}_chunk_{chunk.index}"
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "document_id": document_id,
                "chunk_index": chunk.index,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **{k: str(v) for k, v in chunk.metadata.items()},
            })

        # ChromaDB upserts by default if IDs already exist
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            f"Added {len(chunks)} chunks for document '{document_id}' ({elapsed:.0f}ms)"
        )
        return len(chunks)

    # ── Search ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Semantic search. Returns top-k most relevant chunks.

        Results are sorted by descending relevance (highest score first).
        Score is cosine similarity: 1.0 = identical, 0.0 = orthogonal.
        """
        self._ensure_initialized()
        k = top_k or self._top_k

        if not query.strip():
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count() or k),
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if not results["ids"] or not results["ids"][0]:
            return search_results

        for i, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB returns cosine distance; convert to similarity
            distance = results["distances"][0][i]
            similarity = 1.0 - distance  # cosine distance → similarity

            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            search_results.append(SearchResult(
                text=results["documents"][0][i],
                score=round(similarity, 4),
                chunk_index=int(meta.get("chunk_index", 0)),
                document_id=meta.get("document_id", ""),
                metadata=meta,
            ))

        # Sort by score descending (should already be, but be explicit)
        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results

    # ── Delete ─────────────────────────────────────────────────────────────

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a document. Returns count deleted."""
        self._ensure_initialized()

        # Find all chunks for this document
        existing = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )

        if not existing["ids"]:
            logger.warning(f"No chunks found for document '{document_id}'")
            return 0

        self._collection.delete(ids=existing["ids"])
        count = len(existing["ids"])
        logger.info(f"Deleted {count} chunks for document '{document_id}'")
        return count

    def delete_all(self) -> int:
        """Wipe the entire collection. Returns count deleted."""
        self._ensure_initialized()
        count = self._collection.count()
        if count > 0:
            # Get all IDs and delete
            all_data = self._collection.get(include=[])
            self._collection.delete(ids=all_data["ids"])
            logger.info(f"Deleted all {count} chunks from collection")
        return count

    # ── Info ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of chunks in the collection."""
        self._ensure_initialized()
        return self._collection.count()

    def get_stats(self) -> dict:
        """Collection statistics."""
        self._ensure_initialized()
        total = self._collection.count()

        # Count unique documents
        if total == 0:
            return {"total_chunks": 0, "documents": 0, "collection": self._collection_name}

        all_meta = self._collection.get(include=["metadatas"])
        doc_ids = set()
        for meta in all_meta["metadatas"]:
            doc_id = meta.get("document_id", "")
            if doc_id:
                doc_ids.add(doc_id)

        return {
            "total_chunks": total,
            "documents": len(doc_ids),
            "collection": self._collection_name,
            "embedding_model": self._model_name,
        }

    def list_documents(self) -> list[dict]:
        """List all ingested documents with chunk counts."""
        self._ensure_initialized()
        total = self._collection.count()
        if total == 0:
            return []

        all_meta = self._collection.get(include=["metadatas"])
        doc_info: dict[str, dict] = {}
        for meta in all_meta["metadatas"]:
            doc_id = meta.get("document_id", "unknown")
            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    "document_id": doc_id,
                    "source": meta.get("source", ""),
                    "chunk_count": 0,
                }
            doc_info[doc_id]["chunk_count"] += 1

        return sorted(doc_info.values(), key=lambda d: d["document_id"])
