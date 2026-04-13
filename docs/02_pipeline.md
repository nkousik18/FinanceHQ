# RAG Pipeline — Stage by Stage

## Pipeline Overview

```
PDF Upload
    │
    ▼
[Stage 1] S3 Upload ──────────── raw PDF stored
    │
    ▼
[Stage 2] Text Extraction ─────── PyMuPDF → raw text per page
    │
    ▼
[Stage 3] Validation ──────────── check text quality, flag scanned-only PDFs
    │
    ▼
[Stage 4] Cleaning ────────────── normalize, remove noise
    │
    ▼
[Stage 5] Chunking ────────────── semantic/sliding window chunks
    │
    ▼
[Stage 6] Embedding ───────────── MiniLM-L6-v2 → 384-dim vectors
    │
    ▼
[Stage 7] Index + Store ────────── FAISS index + chunks.json → S3
    │
    ▼
[Stage 8] Query: Retrieve ─────── embed question → FAISS top-k
    │
    ▼
[Stage 9] Prompt Assembly ─────── intent detection → prompt variant (A/B)
    │
    ▼
[Stage 10] LLM Inference ──────── Bytez API → answer
    │
    ▼
[Stage 11] Evaluation ─────────── Bytez evaluator → faithfulness / relevance scores
    │
    ▼
[Stage 12] MLflow Logging ─────── log prompt variant, chunks, scores, latency
```

---

## Stage 1 — S3 Upload

**Input:** Raw PDF (browser upload)  
**Output:** `s3://loandoc-bucket/uploads/{user_id}/{session_id}/original.pdf`

- Django receives the file, calls FastAPI `/upload`
- FastAPI writes directly to S3 using `boto3`
- Returns `session_id` (format: `{user_id}_{timestamp}`)
- All subsequent pipeline stages reference this `session_id`

---

## Stage 2 — Text Extraction

**Tool:** PyMuPDF (`fitz`)  
**Input:** S3 PDF path  
**Output:** `s3://loandoc-bucket/extracted/{session_id}/raw_text.txt`

```python
# Pseudocode
doc = fitz.open(pdf_bytes)
pages = [page.get_text() for page in doc]
raw_text = "\n\n".join(f"=== PAGE {i+1} ===\n{text}" for i, text in enumerate(pages))
```

**Handles:** standard PDFs, text-layer PDFs, mixed content  
**Does NOT handle:** pure scanned image PDFs (flagged in validation stage)

---

## Stage 3 — Validation

**Input:** `raw_text.txt`  
**Output:** `s3://loandoc-bucket/extracted/{session_id}/validation_report.json`

Checks performed:

| Check | Condition | Action |
|---|---|---|
| Minimum content | `len(text) < 500 chars` | Fail — likely scanned or empty |
| Language detection | Not English | Warn (still process) |
| Gibberish ratio | `>30% non-alphanumeric` | Fail — likely garbled OCR |
| Page count | `0 pages extracted` | Fail — PDF unreadable |
| Loan document signal | None of `[loan, interest, borrower, lender, payment, principal]` found | Warn — may not be a loan doc |

Validation result schema:
```json
{
  "session_id": "user54_20250101_120000",
  "passed": true,
  "warnings": ["language: es detected"],
  "errors": [],
  "char_count": 14200,
  "page_count": 8
}
```

If `passed: false` → pipeline halts, session status set to `FAILED`, user notified.

---

## Stage 4 — Cleaning

**Input:** `raw_text.txt`  
**Output:** `s3://loandoc-bucket/extracted/{session_id}/cleaned_text.txt`

Transformations applied (in order):

1. **Page header/footer removal** — strip repeating lines (page numbers, doc titles that repeat)
2. **Whitespace normalization** — collapse multiple spaces/newlines
3. **Special character handling** — preserve `%`, `$`, `.`, `,` (critical for loan docs); strip decorative characters
4. **Table structure preservation** — detect pipe/tab-aligned tables, reformat as readable prose
5. **Hyphenation fix** — rejoin words broken across lines (`mort-\ngage` → `mortgage`)
6. **Encoding normalization** — replace smart quotes, em-dashes with ASCII equivalents

What is NOT removed:
- Numbers (interest rates, amounts, dates — all critical)
- Legal clause markers (Section 4.2, Clause 7)
- Monetary symbols

---

## Stage 5 — Chunking

**Input:** `cleaned_text.txt`  
**Output:** `s3://loandoc-bucket/chunks/{session_id}/chunks.json`

**Strategy:** Sliding window with sentence boundary respect

Parameters (configurable, tracked in MLflow):

| Parameter | Default | Notes |
|---|---|---|
| `chunk_size` | 400 tokens | ~300 words |
| `chunk_overlap` | 80 tokens | Preserves context at boundaries |
| `min_chunk_size` | 50 tokens | Discard fragments |
| `split_by` | sentence boundary | Never cut mid-sentence |

```json
// chunks.json schema
[
  {
    "chunk_id": 0,
    "text": "The borrower agrees to repay...",
    "page_ref": 2,
    "char_start": 1200,
    "char_end": 1800,
    "token_count": 112
  }
]
```

**Why not semantic chunking?** Semantic chunking (embedding-based) requires a model call per boundary decision — expensive at this scale. Sliding window with sentence boundaries is fast, deterministic, and good enough for structured financial docs.

---

## Stage 6 — Embedding

**Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Dimensions:** 384  
**Runs:** locally inside the FastAPI container (no API call)  
**Output:** `s3://loandoc-bucket/chunks/{session_id}/chunk_embeddings.npy`

```python
# Shape: (num_chunks, 384)
# Stored as numpy .npy file, loaded with np.load()
```

Cost: $0. MiniLM runs in CPU in ~100ms for a 50-chunk document.

---

## Stage 7 — Index + Store

**Tool:** FAISS (`IndexFlatIP` — inner product / cosine similarity with normalized vectors)

1. Load `chunk_embeddings.npy`
2. Normalize each vector: `v / ||v||`
3. Build `faiss.IndexFlatIP`
4. Serialize index to bytes
5. Upload to `s3://loandoc-bucket/chunks/{session_id}/faiss.index`

At query time: download `faiss.index` and `chunks.json` from S3, load into RAM, search.

**In-memory cache:** once loaded for a session, the index is cached in the FastAPI process for the session lifetime to avoid repeated S3 downloads.

---

## Stage 8 — Retrieval

**Input:** User question (string)  
**Output:** Top-k chunks with similarity scores

```python
q_vec = embed(question)                  # MiniLM, 384-dim
q_vec = q_vec / np.linalg.norm(q_vec)   # normalize
scores, indices = index.search(q_vec, k=5)
chunks = [chunks_list[i] for i in indices[0]]
```

Default `k=5`. Configurable per prompt variant (tracked in MLflow).

---

## Stage 9 — Prompt Assembly

See `07_prompt_engineering.md` for full details.

Short version:
1. Detect intent from question (rule-based: finance / summary / explanation / retrieval)
2. Select prompt template (A/B variant selection)
3. Assemble: `[system instruction] + [context chunks] + [question]`
4. Trim to max token budget (prevent Bytez API cost overrun)

---

## Stage 10 — LLM Inference

See `08_llm_bytez.md` for full details.

Short version:
- POST assembled prompt to Bytez API
- Stream tokens back to FastAPI
- FastAPI streams to Django via SSE
- Django renders tokens progressively in chat UI

---

## Stage 11 — Evaluation

See `09_evaluation_abtesting.md` for full details.

Short version:
- After answer is generated, call Bytez evaluator model
- Scores: faithfulness (is it grounded in retrieved chunks?), relevance (does it answer the question?)
- Scores stored in MLflow alongside the run

---

## Stage 12 — MLflow Logging

See `10_mlflow_tracking.md` for full details.

Every query creates one MLflow run logging:
- Prompt variant used
- Number of chunks retrieved
- `k` value used
- Bytez model used
- Latency (retrieval, inference, eval)
- Faithfulness score
- Relevance score
- Token count (input + output)
