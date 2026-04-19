"""Tests for the ChromaDB vector store.

Covers:
  - Ingest 20 chunks, verify they're stored
  - Semantic search returns relevant results
  - Top-k parameter controls result count
  - Score ordering (most relevant first)
  - Delete by document ID
  - Collection stats and document listing
  - Full ingest-to-search pipeline
"""

import os
import tempfile
import shutil
from pathlib import Path

import pytest

from src.rag.chunker import Chunk, chunk_text
from src.rag.vectorstore import VectorStore, SearchResult


@pytest.fixture
def tmp_persist_dir():
    """Create a temporary directory for ChromaDB persistence."""
    d = tempfile.mkdtemp(prefix="chromadb_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_persist_dir):
    """A fresh VectorStore backed by a temp directory."""
    config = {
        "rag": {
            "collection_name": "test_collection",
            "top_k": 5,
            "persist_directory": tmp_persist_dir,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        }
    }
    return VectorStore(config)


def _make_chunks(texts: list[str], doc_id_prefix: str = "test") -> list[Chunk]:
    """Helper to create Chunk objects from a list of texts."""
    return [
        Chunk(
            text=text,
            index=i,
            token_count=len(text.split()),
            start_char=0,
            end_char=len(text),
            metadata={"source": f"{doc_id_prefix}.txt"},
        )
        for i, text in enumerate(texts)
    ]


# ── 20 distinct chunks covering different ML topics ──────────────────────

ML_TOPICS = [
    "Gradient descent is an optimization algorithm that minimizes the loss function by iteratively adjusting model parameters in the direction of steepest descent.",
    "Backpropagation computes gradients of the loss with respect to each weight by applying the chain rule from the output layer backward through the network.",
    "Convolutional neural networks use learnable filters to extract spatial features from images at different scales through convolutional and pooling layers.",
    "Recurrent neural networks process sequential data by maintaining a hidden state that captures information from previous time steps in the sequence.",
    "Transformers use self-attention mechanisms to process all positions in a sequence simultaneously, replacing the sequential processing of RNNs.",
    "BERT is a bidirectional transformer pre-trained on masked language modeling and next sentence prediction tasks using large text corpora.",
    "GPT models are autoregressive transformers trained to predict the next token in a sequence, enabling text generation capabilities.",
    "Transfer learning fine-tunes a pre-trained model on a smaller task-specific dataset, dramatically reducing the amount of labeled data needed.",
    "Batch normalization normalizes layer inputs to stabilize training and allow higher learning rates by reducing internal covariate shift.",
    "Dropout randomly deactivates neurons during training to prevent overfitting by forcing the network to learn redundant representations.",
    "K-means clustering partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroid positions.",
    "Principal component analysis finds orthogonal axes that capture the maximum variance in the data for dimensionality reduction.",
    "Random forests combine multiple decision trees trained on random subsets of data and features to improve prediction accuracy and reduce overfitting.",
    "Support vector machines find the hyperplane that maximizes the margin between classes in the feature space for classification tasks.",
    "Reinforcement learning trains agents to make sequential decisions by maximizing cumulative rewards received from the environment.",
    "Q-learning is a model-free reinforcement learning algorithm that learns the expected utility of taking actions in particular states.",
    "Word embeddings map words to dense vectors where semantically similar words have similar vector representations in continuous space.",
    "Attention mechanisms allow models to focus on relevant parts of the input when producing each element of the output sequence.",
    "Data augmentation artificially increases training set size by applying transformations like rotation, flipping, and color jittering to existing samples.",
    "Cross-validation evaluates model performance by training and testing on different subsets of the data to estimate generalization ability.",
]


class TestVectorStoreAdd:
    def test_add_20_chunks(self, store):
        chunks = _make_chunks(ML_TOPICS)
        assert len(chunks) == 20
        added = store.add_chunks(chunks, document_id="ml_textbook")
        assert added == 20
        assert store.count() == 20

    def test_add_empty_list(self, store):
        added = store.add_chunks([], document_id="empty")
        assert added == 0
        assert store.count() == 0

    def test_upsert_same_document_replaces(self, store):
        chunks = _make_chunks(ML_TOPICS[:5])
        store.add_chunks(chunks, document_id="doc_a")
        assert store.count() == 5

        # Upsert same doc with different content
        new_chunks = _make_chunks(["Updated content for chunk zero."] * 5)
        store.add_chunks(new_chunks, document_id="doc_a")
        assert store.count() == 5  # same IDs, so count stays the same


class TestVectorStoreSearch:
    @pytest.fixture(autouse=True)
    def _seed_data(self, store):
        chunks = _make_chunks(ML_TOPICS)
        store.add_chunks(chunks, document_id="ml_textbook")

    def test_search_returns_results(self, store):
        results = store.search("How does backpropagation work?")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_top_k_limits_results(self, store):
        results_3 = store.search("neural networks", top_k=3)
        results_10 = store.search("neural networks", top_k=10)
        assert len(results_3) == 3
        assert len(results_10) == 10

    def test_relevance_ordering(self, store):
        results = store.search("backpropagation gradient chain rule", top_k=5)
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_relevant_chunk_ranks_high(self, store):
        """The chunk about backpropagation should rank in top-3 for a backprop query."""
        results = store.search("How does backpropagation compute gradients?", top_k=5)
        top_3_texts = " ".join(r.text.lower() for r in results[:3])
        assert "backpropagation" in top_3_texts or "gradient" in top_3_texts

    def test_cnn_query_finds_cnn_chunk(self, store):
        results = store.search("convolutional filters for image recognition", top_k=3)
        top_texts = " ".join(r.text.lower() for r in results)
        assert "convolutional" in top_texts or "filter" in top_texts

    def test_clustering_query_finds_kmeans(self, store):
        results = store.search("how to cluster similar data points together", top_k=3)
        top_texts = " ".join(r.text.lower() for r in results)
        assert "cluster" in top_texts or "k-means" in top_texts

    def test_empty_query_returns_empty(self, store):
        results = store.search("", top_k=5)
        assert results == []

    def test_search_result_has_metadata(self, store):
        results = store.search("transformers", top_k=1)
        assert len(results) == 1
        assert results[0].document_id == "ml_textbook"
        assert results[0].chunk_index >= 0
        assert results[0].score > 0

    def test_default_top_k_from_config(self, store):
        """Config sets top_k=5, so search without explicit top_k returns 5."""
        results = store.search("machine learning")
        assert len(results) == 5


class TestVectorStoreDelete:
    def test_delete_by_document_id(self, store):
        chunks_a = _make_chunks(ML_TOPICS[:10])
        chunks_b = _make_chunks(ML_TOPICS[10:])
        store.add_chunks(chunks_a, document_id="doc_a")
        store.add_chunks(chunks_b, document_id="doc_b")
        assert store.count() == 20

        deleted = store.delete_document("doc_a")
        assert deleted == 10
        assert store.count() == 10

        # Remaining chunks should all be from doc_b
        results = store.search("gradient descent", top_k=10)
        for r in results:
            assert r.document_id == "doc_b"

    def test_delete_nonexistent_returns_zero(self, store):
        deleted = store.delete_document("nonexistent_doc")
        assert deleted == 0

    def test_delete_all(self, store):
        chunks = _make_chunks(ML_TOPICS)
        store.add_chunks(chunks, document_id="ml_textbook")
        assert store.count() == 20

        deleted = store.delete_all()
        assert deleted == 20
        assert store.count() == 0


class TestVectorStoreStats:
    def test_stats_empty_store(self, store):
        stats = store.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["documents"] == 0

    def test_stats_after_ingestion(self, store):
        store.add_chunks(_make_chunks(ML_TOPICS[:10]), document_id="doc_1")
        store.add_chunks(_make_chunks(ML_TOPICS[10:]), document_id="doc_2")

        stats = store.get_stats()
        assert stats["total_chunks"] == 20
        assert stats["documents"] == 2
        assert stats["collection"] == "test_collection"
        assert "MiniLM" in stats["embedding_model"]

    def test_list_documents(self, store):
        store.add_chunks(_make_chunks(ML_TOPICS[:7]), document_id="paper_a")
        store.add_chunks(_make_chunks(ML_TOPICS[7:15]), document_id="paper_b")
        store.add_chunks(_make_chunks(ML_TOPICS[15:]), document_id="paper_c")

        docs = store.list_documents()
        assert len(docs) == 3
        doc_map = {d["document_id"]: d for d in docs}
        assert doc_map["paper_a"]["chunk_count"] == 7
        assert doc_map["paper_b"]["chunk_count"] == 8
        assert doc_map["paper_c"]["chunk_count"] == 5


class TestEndToEndPipeline:
    """Full pipeline: read file → chunk → embed → search."""

    def test_ingest_txt_then_search(self, store):
        from src.rag.ingest import DocumentIngestor

        fixtures = Path(__file__).parent / "fixtures"
        ingestor = DocumentIngestor({"rag": {"chunk_max_tokens": 100, "chunk_overlap_tokens": 15}})
        result = ingestor.ingest(str(fixtures / "sample_10page.txt"))
        assert result.chunk_count >= 3

        store.add_chunks(result.chunks, document_id="ml_book")
        assert store.count() == result.chunk_count

        # Search for a topic that's in the document
        results = store.search("neural networks deep learning", top_k=3)
        assert len(results) == 3
        assert any("neural" in r.text.lower() for r in results)
