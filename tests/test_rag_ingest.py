"""Tests for the RAG document ingestion pipeline.

Covers:
  - Sentence-aware chunking with overlap
  - Token count accuracy
  - Chunk boundary respects sentences
  - PDF extraction
  - TXT/MD extraction
  - File upload via /ingest endpoint
  - Batch ingestion via CLI path
  - Edge cases: empty file, unsupported format, oversized file
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.rag.chunker import chunk_text, Chunk, _split_sentences, _count_tokens
from src.rag.ingest import DocumentIngestor, SUPPORTED_EXTENSIONS

FIXTURES = Path(__file__).parent / "fixtures"


# ── Chunker unit tests ────────────────────────────────────────────────────


class TestChunker:
    def test_basic_chunking_produces_chunks(self):
        text = "Hello world. " * 200  # ~400 tokens
        chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert chunk.token_count <= 120  # some tolerance for sentence packing

    def test_chunk_indices_are_sequential(self):
        text = "This is a sentence. " * 100
        chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_overlap_shares_text_between_chunks(self):
        # Create text with distinct sentences
        sentences = [f"Sentence number {i} has unique content." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=30, overlap_tokens=10)

        assert len(chunks) >= 3

        # Verify overlap: some words from end of chunk N appear at start of chunk N+1
        for i in range(len(chunks) - 1):
            current_words = set(chunks[i].text.split()[-15:])  # last 15 words
            next_words = set(chunks[i + 1].text.split()[:15])  # first 15 words
            overlap = current_words & next_words
            assert len(overlap) > 0, f"No overlap between chunk {i} and {i+1}"

    def test_does_not_break_mid_sentence(self):
        text = (
            "Napoleon was born on the island of Corsica in 1769. "
            "He rose to prominence during the French Revolution. "
            "He became Emperor of France in 1804. "
            "His campaigns reshaped European politics. "
        ) * 20
        chunks = chunk_text(text, max_tokens=40, overlap_tokens=8)

        for chunk in chunks:
            # Every chunk should end with a complete sentence (period)
            stripped = chunk.text.strip()
            assert stripped[-1] in ".!?", f"Chunk {chunk.index} ends mid-sentence: ...{stripped[-30:]}"

    def test_single_sentence_exceeding_max_tokens(self):
        # One giant sentence with no periods
        text = "word " * 600
        chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)
        assert len(chunks) >= 1  # should still produce at least one chunk
        assert chunks[0].token_count > 0

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_metadata_propagates_to_chunks(self):
        text = "This is a test document. It has multiple sentences. Each one is important."
        meta = {"source": "test.txt", "author": "tester"}
        chunks = chunk_text(text, max_tokens=20, overlap_tokens=5, metadata=meta)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["author"] == "tester"

    def test_char_offsets_are_set(self):
        text = "First sentence here. Second sentence here. Third sentence here."
        chunks = chunk_text(text, max_tokens=10, overlap_tokens=3)
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char

    def test_chunk_count_for_known_document(self):
        """A 5000-word doc at 512 tokens/chunk should produce ~10–15 chunks."""
        text = "This is a complete sentence with several words in it. " * 500  # ~5000 words
        chunks = chunk_text(text, max_tokens=512, overlap_tokens=64)
        assert 8 <= len(chunks) <= 20, f"Expected 8-20 chunks, got {len(chunks)}"


class TestSentenceSplitter:
    def test_splits_on_period(self):
        text = "First sentence. Second sentence. Third sentence."
        sents = _split_sentences(text)
        assert len(sents) >= 2

    def test_handles_abbreviations(self):
        text = "Dr. Smith went to Washington. He met Mr. Jones there."
        sents = _split_sentences(text)
        # Should not split on "Dr." or "Mr."
        assert all("Dr" in s or "Smith" in s or "He" in s for s in sents)

    def test_handles_no_punctuation(self):
        text = "This is a text without any punctuation marks"
        sents = _split_sentences(text)
        assert len(sents) >= 1
        assert sents[0].strip() == text.strip()


class TestTokenCounter:
    def test_counts_words(self):
        assert _count_tokens("hello world") == 2
        assert _count_tokens("one two three four five") == 5
        assert _count_tokens("") == 0


# ── Ingestor unit tests ──────────────────────────────────────────────────


class TestDocumentIngestor:
    def test_ingest_txt_file(self):
        ingestor = DocumentIngestor()
        result = ingestor.ingest(str(FIXTURES / "sample_10page.txt"))
        assert result.chunk_count > 0
        assert result.char_count > 1000
        assert result.format == "txt"
        assert result.filename == "sample_10page.txt"
        assert result.duration_ms > 0
        assert not result.errors

    def test_ingest_md_file(self):
        ingestor = DocumentIngestor()
        result = ingestor.ingest(str(FIXTURES / "sample_short.md"))
        assert result.chunk_count >= 1
        assert result.format == "md"
        assert "API Reference" in result.chunks[0].text

    def test_ingest_pdf_file(self):
        pdf_path = FIXTURES / "sample_10page.pdf"
        if not pdf_path.exists():
            pytest.skip("Test PDF not found")

        ingestor = DocumentIngestor()
        result = ingestor.ingest(str(pdf_path))
        assert result.chunk_count > 0
        assert result.format == "pdf"
        # Should extract text from all 10 pages
        all_text = " ".join(c.text for c in result.chunks)
        assert "machine learning" in all_text.lower() or "neural" in all_text.lower()

    def test_10page_doc_produces_reasonable_chunks(self):
        """A 10-page document (~10K chars, ~1800 words) at 512 tokens/chunk."""
        ingestor = DocumentIngestor({"rag": {"chunk_max_tokens": 512, "chunk_overlap_tokens": 64}})
        result = ingestor.ingest(str(FIXTURES / "sample_10page.txt"))
        assert result.chunk_count >= 3, f"Expected ≥3 chunks, got {result.chunk_count}"
        # Verify no chunk massively exceeds the limit
        for c in result.chunks:
            assert c.token_count <= 600, f"Chunk {c.index} has {c.token_count} tokens (max 512+buffer)"

    def test_chunk_overlap_is_present(self):
        ingestor = DocumentIngestor({"rag": {"chunk_max_tokens": 100, "chunk_overlap_tokens": 20}})
        result = ingestor.ingest(str(FIXTURES / "sample_10page.txt"))

        if result.chunk_count < 3:
            pytest.skip("Not enough chunks to test overlap")

        # Check overlap between consecutive chunks
        for i in range(min(3, result.chunk_count - 1)):
            c1_words = set(result.chunks[i].text.split()[-20:])
            c2_words = set(result.chunks[i + 1].text.split()[:20])
            overlap = c1_words & c2_words
            assert len(overlap) > 0, f"No overlap between chunks {i} and {i+1}"

    def test_rejects_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            f.write(b"fake excel content")
            f.flush()
            try:
                ingestor = DocumentIngestor()
                with pytest.raises(ValueError, match="Unsupported format"):
                    ingestor.ingest(f.name)
            finally:
                os.unlink(f.name)

    def test_rejects_missing_file(self):
        ingestor = DocumentIngestor()
        with pytest.raises(FileNotFoundError):
            ingestor.ingest("/nonexistent/path/doc.txt")

    def test_handles_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("")
            f.flush()
            try:
                ingestor = DocumentIngestor()
                result = ingestor.ingest(f.name)
                assert result.chunk_count == 0
                assert "no_text_extracted" in result.errors
            finally:
                os.unlink(f.name)

    def test_batch_ingestion(self):
        ingestor = DocumentIngestor()
        results = ingestor.ingest_directory(str(FIXTURES))
        assert len(results) >= 2  # at least sample_10page.txt and sample_short.md
        total_chunks = sum(r.chunk_count for r in results)
        assert total_chunks > 0

    def test_config_overrides(self):
        """Custom chunk size should produce more/fewer chunks."""
        small_ingestor = DocumentIngestor({"rag": {"chunk_max_tokens": 50, "chunk_overlap_tokens": 10}})
        large_ingestor = DocumentIngestor({"rag": {"chunk_max_tokens": 1000, "chunk_overlap_tokens": 100}})

        small_result = small_ingestor.ingest(str(FIXTURES / "sample_10page.txt"))
        large_result = large_ingestor.ingest(str(FIXTURES / "sample_10page.txt"))

        assert small_result.chunk_count > large_result.chunk_count


# ── API endpoint tests ───────────────────────────────────────────────────


class TestIngestEndpoint:
    @pytest.fixture
    def client(self):
        os.environ["LLM_MOCK_MODE"] = "true"
        os.environ.pop("CHATBOT_API_KEY", None)
        from src.config import load_config
        load_config.cache_clear()
        from src.api.main import app
        with TestClient(app) as c:
            yield c

    def test_upload_txt_file(self, client):
        content = "This is a test document. It has several sentences. Each one matters for chunking."
        resp = client.post(
            "/ingest",
            files={"file": ("test.txt", content.encode(), "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.txt"
        assert data["chunk_count"] >= 1
        assert data["char_count"] > 0
        assert len(data["chunks"]) == data["chunk_count"]
        assert "text_preview" in data["chunks"][0]

    def test_upload_md_file(self, client):
        content = "# Title\n\nThis is markdown content. It has headers and paragraphs. Testing the ingestion pipeline."
        resp = client.post(
            "/ingest",
            files={"file": ("doc.md", content.encode(), "text/markdown")},
        )
        assert resp.status_code == 200
        assert resp.json()["format"] == "md"

    def test_rejects_unsupported_format(self, client):
        resp = client.post(
            "/ingest",
            files={"file": ("data.csv", b"a,b,c\n1,2,3", "text/csv")},
        )
        assert resp.status_code == 400
        assert "Unsupported format" in resp.json()["detail"]

    def test_upload_returns_chunk_previews(self, client):
        text = ("Sentence about machine learning. " * 50)
        resp = client.post(
            "/ingest",
            files={"file": ("ml_doc.txt", text.encode(), "text/plain")},
        )
        data = resp.json()
        for chunk in data["chunks"]:
            assert "index" in chunk
            assert "token_count" in chunk
            assert len(chunk["text_preview"]) <= 200
