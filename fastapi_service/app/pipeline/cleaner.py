"""
Cleaner for Textract extraction output.

Handles known artifacts from loan PDF extraction:
  - Numbers with spaces:       "110 000"  → "110000"
  - Indian lakh formatting:    "55 00 000" → "5500000"
  - Digit-letter concat:       "36MONTHS"  → "36 MONTHS"
  - Repeated page headers:     "Shinhan Bank" on every page → stripped from text
  - Empty checkbox form fields: Male/Female/Married fields with no value → removed
  - Duplicate form field keys:  keep highest-confidence non-empty value
  - Truncated values:           flagged, not removed (LLM can note uncertainty)

Input:  ExtractionResult (from extractor.py)
Output: CleanedResult
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.pipeline.extractor import ExtractionResult, FormField, Table, PageResult
from app.core.logging import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Known page-level noise patterns (repeated headers/footers/artifacts)
# ------------------------------------------------------------------

_HEADER_PATTERNS = [
    re.compile(r"^S$"),                        # logo artifact
    re.compile(r"^Shinhan Bank$", re.I),       # repeated bank name
    re.compile(r"^India$", re.I),              # repeated country name
    re.compile(r"^Extend Your Financial Network", re.I),  # tagline
]

# Form field keys that are checkbox options — if value is empty they're noise
_CHECKBOX_KEYS = {
    "Male", "Female", "Transgender",
    "Single", "Married", "Divorced", "Widow",
    "Mr.", "Ms.", "M/s.", "Others",
    "S-Service", "B-Business", "O-Others",
    "Private Sector", "Public Sector", "Government Sector",
    "Professional", "Self Employed", "Retired", "Housewife",
    "Politician", "X-Not Categorised",
    "In-India", "Children",
    "Yes", "No",
    "Guarantor", "Photo",
}

# Patterns that suggest a value is truncated
_TRUNCATED_PATTERNS = [
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,3}$"),   # date missing last digit: 15/02/200
    re.compile(r"[a-zA-Z]$"),                    # URL cut off mid-word
]


# ------------------------------------------------------------------
# Output dataclass
# ------------------------------------------------------------------

@dataclass
class CleanedFormField:
    key: str
    value: str
    page: int
    confidence: float
    is_truncated: bool = False
    original_value: Optional[str] = None   # set if value was modified


@dataclass
class CleanedTable:
    page: int
    rows: int
    cols: int
    data: list[list[str]] = field(default_factory=list)   # [row][col] → text

    def to_markdown(self) -> str:
        if not self.data:
            return ""
        header = self.data[0]
        sep = ["---"] * len(header)
        rows = self.data[1:]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


@dataclass
class CleanedPage:
    page_number: int
    text: str
    avg_confidence: float
    has_handwriting: bool
    lines_removed: int = 0   # header/footer lines stripped


@dataclass
class CleanedResult:
    session_id: str
    pages: list[CleanedPage] = field(default_factory=list)
    form_fields: list[CleanedFormField] = field(default_factory=list)
    tables: list[CleanedTable] = field(default_factory=list)
    total_pages: int = 0
    overall_avg_confidence: float = 0.0
    has_handwriting: bool = False
    cleaning_notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Number cleaning
# ------------------------------------------------------------------

def _clean_number(value: str) -> str:
    """
    Fix spaces inside numeric strings produced by handwriting OCR.
    Examples:
        "110 000"   → "110000"
        "55 00 000" → "5500000"
        "50 000"    → "50000"
        "15 000."   → "15000"
        "1,50,000"  → "1,50,000"  (Indian format — keep commas)
        "7,50,000"  → "7,50,000"  (valid, leave alone)
    """
    # Only process strings that look like numbers with spaces (digits + spaces + optional punctuation)
    if re.fullmatch(r"[\d\s,\.]+", value):
        # Remove trailing punctuation
        value = value.rstrip(".")
        # Remove spaces between digits
        value = re.sub(r"(\d)\s+(\d)", lambda m: m.group(1) + m.group(2), value)
        # Re-run to catch chains like "55 00 000"
        value = re.sub(r"(\d)\s+(\d)", lambda m: m.group(1) + m.group(2), value)
    return value.strip()


def _clean_digit_letter_concat(value: str) -> str:
    """
    Add space between a digit and an uppercase letter run.
    "36MONTHS" → "36 MONTHS"
    "3200PaLAKH" → "3200 PaLAKH"
    """
    return re.sub(r"(\d)([A-Z][a-zA-Z])", r"\1 \2", value)


def _is_truncated(value: str) -> bool:
    for pattern in _TRUNCATED_PATTERNS:
        if pattern.search(value):
            return True
    return False


def _clean_value(raw: str) -> tuple[str, Optional[str]]:
    """
    Apply all value cleaning steps.
    Returns (cleaned_value, original_value_if_changed).
    """
    original = raw
    v = raw.strip()
    v = _clean_number(v)
    v = _clean_digit_letter_concat(v)
    changed = v != original
    return v, (original if changed else None)


# ------------------------------------------------------------------
# Text line cleaning
# ------------------------------------------------------------------

def _is_header_noise(line: str) -> bool:
    line = line.strip()
    for pattern in _HEADER_PATTERNS:
        if pattern.match(line):
            return True
    return False


def _clean_page_text(page: PageResult) -> CleanedPage:
    lines = page.text.split("\n")
    kept = []
    removed = 0
    for line in lines:
        if _is_header_noise(line):
            removed += 1
        else:
            kept.append(line)

    return CleanedPage(
        page_number=page.page_number,
        text="\n".join(kept).strip(),
        avg_confidence=page.avg_confidence,
        has_handwriting=page.has_handwriting,
        lines_removed=removed,
    )


# ------------------------------------------------------------------
# Form field cleaning
# ------------------------------------------------------------------

def _clean_form_fields(fields: list[FormField]) -> list[CleanedFormField]:
    """
    1. Drop pure checkbox noise (known checkbox key + empty value)
    2. Clean values
    3. Deduplicate: same key on same page → keep highest-value non-empty entry
    """
    cleaned: list[CleanedFormField] = []

    for f in fields:
        key = f.key.strip()
        raw_value = f.value.strip()

        # Drop empty checkbox fields
        if not raw_value and key in _CHECKBOX_KEYS:
            continue

        clean_val, original = _clean_value(raw_value)
        truncated = _is_truncated(clean_val)

        cleaned.append(CleanedFormField(
            key=key,
            value=clean_val,
            page=f.page,
            confidence=f.value_confidence,
            is_truncated=truncated,
            original_value=original,
        ))

    # Deduplicate: (key, page) → keep non-empty, then highest confidence
    seen: dict[tuple[str, int], CleanedFormField] = {}
    for f in cleaned:
        k = (f.key, f.page)
        if k not in seen:
            seen[k] = f
        else:
            existing = seen[k]
            # Prefer non-empty over empty
            if not existing.value and f.value:
                seen[k] = f
            # If both non-empty, prefer higher confidence
            elif f.value and f.confidence > existing.confidence:
                seen[k] = f

    return sorted(seen.values(), key=lambda x: (x.page, x.key))


# ------------------------------------------------------------------
# Table cleaning
# ------------------------------------------------------------------

def _clean_tables(tables: list[Table]) -> list[CleanedTable]:
    cleaned = []
    for t in tables:
        grid: dict[tuple[int, int], str] = {
            (c.row, c.col): _clean_value(c.text)[0]
            for c in t.cells
        }
        data = []
        for r in range(1, t.rows + 1):
            row = [grid.get((r, c), "") for c in range(1, t.cols + 1)]
            data.append(row)

        cleaned.append(CleanedTable(
            page=t.page,
            rows=t.rows,
            cols=t.cols,
            data=data,
        ))
    return cleaned


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def clean_extraction(result: ExtractionResult) -> CleanedResult:
    """
    Clean an ExtractionResult and return a CleanedResult.
    Does not modify the input.
    """
    logger.info("cleaning_start", session_id=result.session_id,
                pages=result.total_pages, form_fields=len(result.form_fields),
                tables=len(result.tables))

    cleaned_pages = [_clean_page_text(p) for p in result.pages]
    cleaned_fields = _clean_form_fields(result.form_fields)
    cleaned_tables = _clean_tables(result.tables)

    notes: list[str] = []
    total_removed = sum(p.lines_removed for p in cleaned_pages)
    if total_removed:
        notes.append(f"Removed {total_removed} repeated header/footer lines across all pages.")

    truncated = [f for f in cleaned_fields if f.is_truncated]
    if truncated:
        notes.append(
            f"{len(truncated)} field(s) appear truncated: "
            + ", ".join(f"{f.key} (p{f.page})" for f in truncated)
        )

    modified = [f for f in cleaned_fields if f.original_value is not None]
    if modified:
        notes.append(f"{len(modified)} field value(s) normalised (number spacing, digit-letter concat).")

    logger.info(
        "cleaning_complete",
        session_id=result.session_id,
        fields_before=len(result.form_fields),
        fields_after=len(cleaned_fields),
        header_lines_removed=total_removed,
        truncated_fields=len(truncated),
        normalised_fields=len(modified),
    )

    return CleanedResult(
        session_id=result.session_id,
        pages=cleaned_pages,
        form_fields=cleaned_fields,
        tables=cleaned_tables,
        total_pages=result.total_pages,
        overall_avg_confidence=result.overall_avg_confidence,
        has_handwriting=result.has_handwriting,
        cleaning_notes=notes,
    )
