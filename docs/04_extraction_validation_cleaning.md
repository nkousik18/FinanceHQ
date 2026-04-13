# Extraction, Validation, and Cleaning

## Extraction — PyMuPDF

### Why PyMuPDF

| Tool | Cost | Handles scanned PDFs | Tables | Speed |
|---|---|---|---|---|
| AWS Textract | $1.50/1000 pages | Yes (OCR) | Yes | Slow (async) |
| PyMuPDF | Free | No (text layer only) | Partial | ~50ms |
| pdfplumber | Free | No | Better than PyMuPDF | ~200ms |
| Surya | Free | Yes (neural OCR) | Yes | ~2-5s |

**Decision:** PyMuPDF for standard PDFs. Loan documents in digital form (e-signed, digital originals) always have a text layer — this covers 95% of real-world use. If a document fails the text content validation, the user is told to upload a clearer PDF.

### Extraction Code Contract

```python
def extract_text_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Returns:
    {
        "pages": [{"page_num": 1, "text": "..."}, ...],
        "full_text": "...",
        "page_count": 8,
        "char_count": 14200,
        "metadata": {"title": "...", "author": "..."}
    }
    Raises: ExtractionError if PDF is unreadable
    """
```

### Page Markers

Raw text preserves page boundaries:
```
=== PAGE 1 ===
LOAN AGREEMENT

This agreement is made on January 15, 2025...

=== PAGE 2 ===
SECTION 2 — REPAYMENT TERMS
...
```

Page markers are stripped before chunking but kept in `raw_text.txt` for debugging.

---

## Validation

### Purpose

Catch bad inputs *early* before expensive operations (embedding, Bytez API calls) are wasted on garbage.

### Validation Checks

#### 1. Minimum Character Count
```python
MINIMUM_CHARS = 500

if len(full_text.strip()) < MINIMUM_CHARS:
    fail("Document appears to be empty or unreadable. "
         "Scanned PDFs without a text layer cannot be processed.")
```

#### 2. Gibberish Ratio
```python
# Ratio of non-printable / non-alphanumeric chars (excluding spaces, punctuation)
GIBBERISH_THRESHOLD = 0.35

printable = sum(c.isalnum() or c in ' .,!?-:;()%$\n' for c in text)
ratio = 1 - (printable / len(text))
if ratio > GIBBERISH_THRESHOLD:
    fail("Document content appears garbled. Check if the PDF has proper text encoding.")
```

#### 3. Page Count
```python
if page_count == 0:
    fail("No pages could be extracted from this PDF.")
```

#### 4. Loan Document Signal (Warning Only)
```python
LOAN_KEYWORDS = [
    "loan", "borrower", "lender", "interest", "principal",
    "repayment", "mortgage", "collateral", "emi", "amortization"
]

found = [kw for kw in LOAN_KEYWORDS if kw in full_text.lower()]
if len(found) < 2:
    warn("This document may not be a loan agreement. "
         "Results may be less accurate.")
```

This is a warning, not a failure — the system still processes it.

#### 5. Language Detection (Warning Only)
Uses `langdetect` library. Non-English documents are processed but flagged:
```python
from langdetect import detect
lang = detect(full_text[:2000])  # sample first 2000 chars
if lang != 'en':
    warn(f"Document appears to be in '{lang}'. "
         f"Results may be less accurate for non-English documents.")
```

### Validation Report

Written to `s3://loandoc-bucket/extracted/{session_id}/validation_report.json`:
```json
{
  "session_id": "...",
  "timestamp": "2025-01-15T14:23:05Z",
  "passed": true,
  "errors": [],
  "warnings": ["document language detected: es"],
  "stats": {
    "char_count": 14200,
    "page_count": 8,
    "gibberish_ratio": 0.02,
    "loan_keywords_found": ["loan", "borrower", "interest", "repayment"]
  }
}
```

---

## Cleaning

### Purpose

Remove noise introduced by PDF formatting that would degrade retrieval quality. A chunk that says "Page 4 of 12 — CONFIDENTIAL" contributes nothing to answering "What is the interest rate?"

### Transformations (Applied in Order)

#### 1. Repeated Line Removal (Headers/Footers)
```python
# Lines that appear on 3+ pages are likely headers/footers
from collections import Counter

all_lines = full_text.split('\n')
line_freq = Counter(line.strip() for line in all_lines if line.strip())
repeated = {line for line, count in line_freq.items() if count >= 3}
cleaned = [line for line in all_lines if line.strip() not in repeated]
```

#### 2. Whitespace Normalization
```python
import re
text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 consecutive newlines
text = re.sub(r'[ \t]{2,}', ' ', text)   # collapse spaces/tabs
text = text.strip()
```

#### 3. Hyphenation Fix
```python
# "mort-\ngage" → "mortgage"
text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
```

#### 4. Smart Quote / Special Character Normalization
```python
replacements = {
    '\u2018': "'", '\u2019': "'",   # smart single quotes
    '\u201c': '"', '\u201d': '"',   # smart double quotes
    '\u2013': '-', '\u2014': '-',   # en-dash, em-dash
    '\u00a0': ' ',                  # non-breaking space
}
for char, replacement in replacements.items():
    text = text.replace(char, replacement)
```

#### 5. Table Reformatting
Simple pipe-separated or whitespace-aligned tables are detected and converted to prose:
```
BEFORE:
Principal  | Rate  | Term
$250,000   | 4.5%  | 30 years

AFTER:
Loan details: Principal $250,000, Rate 4.5%, Term 30 years.
```

This is done with a simple heuristic — lines where `|` appears 2+ times, or where `\t` separates 3+ columns.

#### 6. Page Marker Removal
```python
text = re.sub(r'=== PAGE \d+ ===\n?', '', text)
```

### What Cleaning Does NOT Do

- Does not reorder content
- Does not remove numbers, dates, percentages
- Does not interpret tables into structured data (that would require Textract-level analysis)
- Does not fix OCR errors (none expected since we use text-layer PDFs)
