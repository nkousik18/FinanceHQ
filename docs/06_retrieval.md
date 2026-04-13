# Retrieval

## What Retrieval Does

Given a user question, retrieval finds the chunks from the uploaded document most likely to contain the answer. The LLM only sees these chunks — not the full document.

## Retrieval Flow

```
User question (string)
    │
    ▼
embed_query(question)          → q_vec: shape (384,)
    │
    ▼
faiss_index.search(q_vec, k)   → scores: (k,), indices: (k,)
    │
    ▼
[chunks[i] for i in indices]   → list of top-k chunk dicts
    │
    ▼
context assembly (next stage)
```

## Similarity Search

FAISS `IndexFlatIP` with L2-normalized vectors performs exact cosine similarity:

```
cosine_similarity(a, b) = dot(a/||a||, b/||b||) = dot(a, b)  [when both normalized]
```

This is the correct metric for sentence embedding similarity. Score range: -1 to 1. In practice, relevant chunks score 0.4–0.8 for loan document queries.

## Top-K Parameter

Default `k=5`. This is tunable per prompt variant and tracked in MLflow as a hyperparameter.

Trade-offs:

| k | Context tokens used | Risk of irrelevant chunks |
|---|---|---|
| 3 | ~1,200 | Lower — only highest-scoring chunks |
| 5 | ~2,000 | Moderate |
| 8 | ~3,200 | Higher — later chunks may be noise |
| 10 | ~4,000 | High — may dilute focused answers |

More context is not always better. A finance question about "prepayment penalty" is best answered with 3 highly relevant chunks than 8 where 5 are about unrelated clauses.

## Score Threshold

A minimum score threshold filters out low-quality retrievals:

```python
SCORE_THRESHOLD = 0.25

chunks = [
    {"text": chunks[i]["text"], "score": float(scores[j]), "chunk_id": int(indices[j])}
    for j, i in enumerate(indices[0])
    if scores[0][j] >= SCORE_THRESHOLD
]
```

If no chunks meet the threshold, the system returns a "no relevant content found" response rather than hallucinating from an empty or irrelevant context.

## Retrieval Result Schema

```json
[
  {
    "chunk_id": 12,
    "text": "The borrower shall repay the principal amount...",
    "score": 0.71,
    "rank": 1
  },
  {
    "chunk_id": 8,
    "text": "In the event of early repayment...",
    "score": 0.54,
    "rank": 2
  }
]
```

This schema is:
- Passed to the prompt builder (text fields)
- Logged to MLflow (scores, chunk_ids)
- Returned to the frontend alongside the answer (for "sources" display)

## Context Assembly

Retrieved chunks are assembled into a context block:

```python
def assemble_context(chunks: list[dict], max_chars: int = 7000) -> str:
    context_parts = []
    total = 0
    for chunk in chunks:
        text = chunk["text"]
        if total + len(text) > max_chars:
            break
        context_parts.append(f"[Source {chunk['rank']}]\n{text}")
        total += len(text)
    return "\n\n".join(context_parts)
```

`max_chars=7000` maps to roughly 1,750 tokens — well within Bytez API context limits while leaving room for the question and system instruction.

## No Relevant Content Case

```python
if not chunks:
    return {
        "answer": "I could not find relevant information in this document to answer your question. "
                  "Please verify the document contains the relevant section.",
        "chunks_used": [],
        "intent": "unknown"
    }
```

This is explicit and honest — better than the LLM guessing from nothing.

## What This Does NOT Do

- **No reranking.** A cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM`) would improve precision but adds latency and complexity. Out of scope for v1.
- **No hybrid search.** BM25 keyword search combined with vector search improves recall for exact term matching (e.g., specific clause numbers). Possible future addition.
- **No multi-document retrieval.** Queries are scoped to one session/document. Cross-document search is not supported.
