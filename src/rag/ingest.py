"""Document ingestion pipeline.

Reads PDF, TXT, and Markdown files, extracts text, chunks with overlap,
and returns structured Chunk objects ready for embedding and indexing.

Supported formats:
  .txt, .md, .markdown  — read as UTF-8 text
  .pdf                   — extracted via PyMuPDF (fitz) or pdfplumber fallback

The pipeline is intentionally synchronous — ingestion is a batch operation
that runs at deploy time or via CLI, not on the hot path.

Usage:
    from src.rag.ingest import DocumentIngestor

    ingestor = DocumentIngestor(config)
    result = ingestor.ingest("/path/to/document.pdf")
    print(f"{result.filename}: {result.chunk_count} chunks")

CLI:
    python -m src.rag.ingest /path/to/file.pdf
    python -m src.rag.ingest /path/to/docs/   # batch: all files in directory
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.rag.chunker import Chunk, chunk_text
from src.monitoring.logging import logger


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


@dataclass
class IngestResult:
    filename: str
    file_path: str
    file_size_bytes: int
    format: str
    char_count: int
    chunk_count: int
    chunks: list[Chunk]
    duration_ms: float
    errors: list[str] = field(default_factory=list)


class DocumentIngestor:
    """Reads documents and splits them into overlapping chunks."""

    def __init__(self, config: dict | None = None):
        rag_cfg = (config or {}).get("rag", {})
        self._max_tokens = rag_cfg.get("chunk_max_tokens", 512)
        self._overlap_tokens = rag_cfg.get("chunk_overlap_tokens", 64)
        self._max_file_size_mb = rag_cfg.get("max_file_size_mb", 50)

    def ingest(self, file_path: str) -> IngestResult:
        """Ingest a single file. Returns IngestResult with chunks."""
        path = Path(file_path)
        start = time.monotonic()
        errors: list[str] = []

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        file_size = path.stat().st_size
        if file_size > self._max_file_size_mb * 1024 * 1024:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB "
                f"(max: {self._max_file_size_mb}MB)"
            )

        # Extract text
        ext = path.suffix.lower()
        if ext == ".pdf":
            text, extract_errors = self._extract_pdf(path)
            errors.extend(extract_errors)
        else:
            text = self._extract_text(path)

        if not text.strip():
            errors.append("no_text_extracted")
            logger.warning(f"No text extracted from {path.name}")

        # Chunk
        metadata = {
            "source": path.name,
            "file_path": str(path.absolute()),
            "format": ext.lstrip("."),
        }
        chunks = chunk_text(
            text,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
            metadata=metadata,
        )

        duration_ms = (time.monotonic() - start) * 1000

        logger.info(
            f"Ingested {path.name}: {len(text)} chars → {len(chunks)} chunks "
            f"({duration_ms:.0f}ms)",
            extra={"endpoint": "/ingest"},
        )

        return IngestResult(
            filename=path.name,
            file_path=str(path.absolute()),
            file_size_bytes=file_size,
            format=ext.lstrip("."),
            char_count=len(text),
            chunk_count=len(chunks),
            chunks=chunks,
            duration_ms=round(duration_ms, 2),
            errors=errors,
        )

    def ingest_directory(self, dir_path: str) -> list[IngestResult]:
        """Ingest all supported files in a directory (non-recursive)."""
        directory = Path(dir_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        results = []
        files = sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not files:
            logger.warning(f"No supported files found in {dir_path}")
            return results

        for file_path in files:
            try:
                result = self.ingest(str(file_path))
                results.append(result)
            except Exception as exc:
                logger.error(f"Failed to ingest {file_path.name}: {exc}")
                results.append(
                    IngestResult(
                        filename=file_path.name,
                        file_path=str(file_path),
                        file_size_bytes=file_path.stat().st_size,
                        format=file_path.suffix.lstrip("."),
                        char_count=0,
                        chunk_count=0,
                        chunks=[],
                        duration_ms=0.0,
                        errors=[str(exc)],
                    )
                )

        total_chunks = sum(r.chunk_count for r in results)
        logger.info(
            f"Batch ingestion: {len(results)} files, {total_chunks} total chunks"
        )
        return results

    @staticmethod
    def _extract_text(path: Path) -> str:
        """Read plain text or markdown file."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                return path.read_text(encoding=enc)
            except (UnicodeDecodeError, ValueError):
                continue
        raise ValueError(f"Could not decode {path.name} with any supported encoding")

    @staticmethod
    def _extract_pdf(path: Path) -> tuple[str, list[str]]:
        """Extract text from PDF. Tries PyMuPDF first, then pdfplumber."""
        errors: list[str] = []

        # Attempt 1: PyMuPDF (fast, handles most PDFs)
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text")
                if page_text.strip():
                    pages.append(page_text)
            doc.close()

            if pages:
                return "\n\n".join(pages), errors
            errors.append("pymupdf_no_text")
        except ImportError:
            errors.append("pymupdf_not_installed")
        except Exception as exc:
            errors.append(f"pymupdf_error:{exc}")

        # Attempt 2: pdfplumber (handles tables and complex layouts better)
        try:
            import pdfplumber

            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append(page_text)

            if pages:
                return "\n\n".join(pages), errors
            errors.append("pdfplumber_no_text")
        except ImportError:
            errors.append("pdfplumber_not_installed")
        except Exception as exc:
            errors.append(f"pdfplumber_error:{exc}")

        # Both failed — return empty with errors
        logger.warning(
            f"PDF text extraction failed for {path.name}. "
            f"Install PyMuPDF (pip install pymupdf) or pdfplumber."
        )
        return "", errors


# ── CLI entry point ────────────────────────────────────────────────────────

def main() -> None:
    """CLI: python -m src.rag.ingest <file_or_directory>"""
    if len(sys.argv) < 2:
        print("Usage: python -m src.rag.ingest <file_or_directory>")
        print("Supported formats: .txt, .md, .pdf")
        sys.exit(1)

    target = sys.argv[1]
    # Load config if available
    try:
        from src.config import load_config
        config = load_config()
    except Exception:
        config = {}

    ingestor = DocumentIngestor(config)

    if os.path.isdir(target):
        results = ingestor.ingest_directory(target)
    else:
        results = [ingestor.ingest(target)]

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'File':<30} {'Format':<6} {'Chars':>8} {'Chunks':>7} {'Time':>8}")
    print(f"{'-'*70}")
    for r in results:
        status = "✓" if not r.errors else "⚠"
        print(
            f"{status} {r.filename:<28} {r.format:<6} {r.char_count:>8} "
            f"{r.chunk_count:>7} {r.duration_ms:>6.0f}ms"
        )
        if r.errors:
            for err in r.errors:
                print(f"    └─ {err}")
    print(f"{'='*70}")
    total_chunks = sum(r.chunk_count for r in results)
    print(f"Total: {len(results)} file(s), {total_chunks} chunk(s)")


if __name__ == "__main__":
    main()
