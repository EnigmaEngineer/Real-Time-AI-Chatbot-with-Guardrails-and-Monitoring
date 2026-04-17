"""Sentence-aware text chunking with configurable overlap.

Splits text into chunks that respect sentence boundaries. A naive
fixed-window chunker cuts mid-sentence, which degrades retrieval
quality — the embedding for "Napoleon was born on the island of"
is useless without "Corsica in 1769."

Algorithm:
  1. Split text into sentences (iterative word scanner, handles abbreviations)
  2. Greedily pack sentences into chunks up to `max_tokens`
  3. When a chunk is full, roll back `overlap_tokens` worth of
     sentences into the start of the next chunk

Token counting uses a simple whitespace split (1 token ≈ 1 word).
For production with tiktoken, swap `_count_tokens`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Abbreviations that should NOT trigger a sentence split
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "approx", "inc", "ltd", "co", "dept", "est", "vol", "fig",
}


@dataclass
class Chunk:
    text: str
    index: int  # 0-based position within the document
    token_count: int
    start_char: int  # character offset in original document
    end_char: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(idx={self.index}, tokens={self.token_count}, text='{preview}...')"


def _count_tokens(text: str) -> int:
    """Approximate token count. 1 whitespace-delimited word ≈ 1 token."""
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling abbreviations and decimal numbers.

    Uses an iterative word-by-word scan instead of regex lookbehinds
    (which Python's re module restricts to fixed-width patterns).
    """
    sentences: list[str] = []
    current: list[str] = []
    words = text.split()

    for word in words:
        current.append(word)

        # Check if this word ends with sentence-ending punctuation
        stripped = word.rstrip('"\')')
        if not stripped or stripped[-1] not in ".!?":
            continue

        # Don't split on abbreviations (e.g. "Dr.", "Mr.")
        base = stripped.rstrip(".!?").lower()
        if base in _ABBREVIATIONS:
            continue

        # Don't split on single letters followed by period (initials like "J.")
        if len(base) == 1 and stripped[-1] == ".":
            continue

        # Don't split on decimal numbers (e.g. "3.14")
        if base.replace(".", "").replace(",", "").isdigit():
            continue

        # This looks like a real sentence boundary
        sentence = " ".join(current).strip()
        if sentence:
            sentences.append(sentence)
        current = []

    # Remainder after last sentence boundary
    if current:
        remainder = " ".join(current).strip()
        if remainder:
            if sentences:
                sentences[-1] += " " + remainder
            else:
                sentences.append(remainder)

    if not sentences:
        sentences = [text.strip()] if text.strip() else []

    return sentences


def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks that respect sentence boundaries.

    Args:
        text: The full document text.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Number of tokens to repeat at the start of the next chunk.
        metadata: Optional metadata to attach to each chunk (e.g. filename).

    Returns:
        List of Chunk objects with text, index, token_count, and char offsets.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    base_meta = metadata or {}
    chunks: list[Chunk] = []

    current_sentences: list[str] = []
    current_tokens = 0
    char_cursor = 0

    def _flush(sents: list[str], start_char: int) -> Chunk:
        joined = " ".join(sents)
        tok_count = _count_tokens(joined)
        end_char = start_char + len(joined)
        return Chunk(
            text=joined,
            index=len(chunks),
            token_count=tok_count,
            start_char=start_char,
            end_char=end_char,
            metadata=dict(base_meta),
        )

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        # Edge case: single sentence exceeds max_tokens — take it as-is
        if sent_tokens > max_tokens and not current_sentences:
            chunk = _flush([sent], char_cursor)
            chunks.append(chunk)
            char_cursor += len(sent) + 1
            continue

        # Would adding this sentence exceed the limit?
        if current_tokens + sent_tokens > max_tokens and current_sentences:
            chunk = _flush(current_sentences, char_cursor)
            chunks.append(chunk)

            # Build overlap: walk backward until we have ~overlap_tokens
            overlap_sents: list[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_tok = _count_tokens(s)
                if overlap_count + s_tok > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_count += s_tok

            # Advance char cursor past non-overlapping portion
            non_overlap = " ".join(current_sentences[: len(current_sentences) - len(overlap_sents)])
            char_cursor += len(non_overlap) + (1 if non_overlap else 0)

            current_sentences = list(overlap_sents)
            current_tokens = overlap_count

        current_sentences.append(sent)
        current_tokens += sent_tokens

    # Flush remaining
    if current_sentences:
        chunk = _flush(current_sentences, char_cursor)
        chunks.append(chunk)

    return chunks
