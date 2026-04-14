"""
Chunker — splits cleaned markdown into overlapping text chunks for embedding.

Strategy: sliding window over lines, respecting markdown section boundaries.
- Tries to keep chunks within a section (## heading) where possible
- Overlap carries the last N tokens of the previous chunk into the next
- Each chunk carries metadata: page, section, chunk index

Input:  CleanedResult
Output: list[Chunk]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from app.pipeline.cleaner import CleanedResult
from app.core.logging import get_logger

logger = get_logger(__name__)

# Defaults — tuneable per experiment
DEFAULT_CHUNK_SIZE = 400      # tokens (approx — we use word count as proxy)
DEFAULT_CHUNK_OVERLAP = 60    # tokens of overlap between consecutive chunks
WORDS_PER_TOKEN = 0.75        # rough conversion: 1 token ≈ 0.75 words


@dataclass
class Chunk:
    chunk_id: str          # "{session_id}-{index}"
    session_id: str
    index: int
    text: str
    word_count: int
    page: int | None       # page number if determinable, else None
    section: str           # nearest markdown heading above this chunk
    token_estimate: int    # approximate token count

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _estimate_tokens(word_count: int) -> int:
    return int(word_count / WORDS_PER_TOKEN)


def _extract_section(line: str) -> str | None:
    """Return heading text if line is a markdown heading, else None."""
    match = re.match(r"^#{1,4}\s+(.+)$", line.strip())
    return match.group(1).strip() if match else None


def _extract_page_from_heading(heading: str) -> int | None:
    """
    Parse page number from headings like 'Page 3 — confidence: 98.1%'
    """
    match = re.search(r"Page\s+(\d+)", heading, re.I)
    return int(match.group(1)) if match else None


# ------------------------------------------------------------------
# Core sliding window chunker
# ------------------------------------------------------------------

def _slide(
    lines: list[str],
    chunk_size_words: int,
    overlap_words: int,
) -> list[list[str]]:
    """
    Slide a window over lines producing overlapping groups.
    Each group = one chunk's lines.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    current_words = 0

    for line in lines:
        line_words = _word_count(line)

        # Long single line exceeds chunk size on its own — split at word boundary
        if line_words > chunk_size_words:
            words = line.split()
            for i in range(0, len(words), chunk_size_words):
                sub = " ".join(words[i: i + chunk_size_words])
                current.append(sub)
                current_words += _word_count(sub)
                if current_words >= chunk_size_words:
                    chunks.append(current[:])
                    current, current_words = _build_overlap(current, overlap_words)
            continue

        current.append(line)
        current_words += line_words

        if current_words >= chunk_size_words:
            chunks.append(current[:])
            current, current_words = _build_overlap(current, overlap_words)

    if current:
        chunks.append(current)

    return chunks


def _build_overlap(lines: list[str], overlap_words: int) -> tuple[list[str], int]:
    """
    Return the tail of lines that fits within overlap_words budget.
    Always keeps at least the last line to guarantee non-empty overlap.
    """
    overlap_lines: list[str] = []
    overlap_count = 0
    for line in reversed(lines):
        w = _word_count(line)
        if overlap_count + w > overlap_words and overlap_lines:
            break
        overlap_lines.insert(0, line)
        overlap_count += w
    return overlap_lines, overlap_count


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def chunk_document(
    cleaned: CleanedResult,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Chunk a CleanedResult into overlapping text chunks.

    Uses the full_text rebuilt from cleaned pages + tables.
    Tracks current section heading and page number from markdown headings.
    """
    logger.info(
        "chunking_start",
        session_id=cleaned.session_id,
        pages=cleaned.total_pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # Rebuild full text from cleaned pages (with table sections inline)
    page_sections: list[str] = []
    for page in cleaned.pages:
        section_text = f"### Page {page.page_number}\n\n{page.text}"
        page_sections.append(section_text)

    # Inject table markdown under each page section
    table_by_page: dict[int, list[str]] = {}
    for table in cleaned.tables:
        table_by_page.setdefault(table.page, []).append(table.to_markdown())

    full_lines: list[str] = []
    for page in cleaned.pages:
        full_lines.append(f"### Page {page.page_number}")
        full_lines.append("")
        full_lines.extend(page.text.splitlines())
        if page.page_number in table_by_page:
            for tmd in table_by_page[page.page_number]:
                full_lines.append("\n[TABLE]")
                full_lines.extend(tmd.splitlines())
        full_lines.append("")

    # Also inject form fields as a dedicated section for retrieval
    if cleaned.form_fields:
        full_lines.append("### Form Fields")
        full_lines.append("")
        for f in cleaned.form_fields:
            if f.value:
                full_lines.append(f"{f.key}: {f.value}")
        full_lines.append("")

    # Slide window
    chunk_size_words = int(chunk_size * WORDS_PER_TOKEN)
    overlap_words = int(overlap * WORDS_PER_TOKEN)
    line_groups = _slide(full_lines, chunk_size_words, overlap_words)

    chunks: list[Chunk] = []
    current_section = "Document"
    current_page: int | None = None

    for idx, group in enumerate(line_groups):
        # Detect section + page from headings within the group
        for line in group:
            heading = _extract_section(line)
            if heading:
                page_num = _extract_page_from_heading(heading)
                if page_num is not None:
                    current_page = page_num
                else:
                    current_section = heading

        text = "\n".join(group).strip()
        if not text:
            continue

        wc = _word_count(text)
        chunks.append(Chunk(
            chunk_id=f"{cleaned.session_id}-{idx}",
            session_id=cleaned.session_id,
            index=idx,
            text=text,
            word_count=wc,
            page=current_page,
            section=current_section,
            token_estimate=_estimate_tokens(wc),
        ))

    logger.info(
        "chunking_complete",
        session_id=cleaned.session_id,
        total_chunks=len(chunks),
        avg_words=round(sum(c.word_count for c in chunks) / len(chunks), 1) if chunks else 0,
    )

    return chunks
