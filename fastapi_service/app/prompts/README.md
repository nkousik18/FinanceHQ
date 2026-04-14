# Prompts

Intent-driven prompt templates for the RAG query pipeline. Each intent gets its own template; the router selects and assembles the right one at query time.

---

## Structure

```
app/prompts/
├── router.py              # Selects template, builds context, returns assembled prompt
└── templates/
    ├── lookup.py          # Retrieve a specific value from the document
    ├── calculate.py       # Perform arithmetic using document values
    ├── compare.py         # Compare two or more values from the document
    ├── explain.py         # Explain a term, clause, or section in plain language
    └── summarise.py       # Summarise the full document (also the fallback intent)
```

---

## How it fits in the pipeline

```
User question
     │
     ▼
IntentClassifier          → detects one or more intents (e.g. LOOKUP, CALCULATE)
     │
     ▼
Retriever                 → fetches top-k chunks from FAISS index
     │
     ▼
router.route(...)         → picks template for primary intent, injects context + question
     │
     ▼
RoutedPrompt              → assembled prompt string, ready for LLM
```

---

## `router.py`

### `route(question, intents, chunks) → RoutedPrompt`

Selects the prompt template for the primary intent, builds a context block from retrieved chunks, and fills the template.

**Arguments**

| Param | Type | Description |
|---|---|---|
| `question` | `str` | Raw user question |
| `intents` | `list[ClassifiedIntent]` | Sorted list from `IntentClassifier`. `intents[0]` is always primary. |
| `chunks` | `list[RetrievedChunk]` | Retrieved chunks, sorted by score (rank 1 = best). |

**Returns: `RoutedPrompt`**

| Field | Type | Description |
|---|---|---|
| `prompt` | `str` | Assembled prompt ready to send to the LLM |
| `intent` | `Intent` | Primary intent that drove template selection |
| `variant` | `str` | Template label, e.g. `"lookup_v1"` — used for MLflow A/B tracking |
| `chunks_used` | `int` | Number of chunks included in the context |
| `context_words` | `int` | Word count of the assembled context |

**Context budget**: Chunks are added greedily until `MAX_CONTEXT_WORDS = 1200` is reached. The first chunk is always included regardless of size.

**Multi-intent**: Only the primary intent drives template selection. Secondary intents are logged for MLflow tracking but do not change the prompt.

**Usage**

```python
from app.prompts.router import route
from app.retrieval.intent_classifier import get_intent_classifier
from app.retrieval.retriever import get_retriever

intents = get_intent_classifier().classify(question)
chunks  = get_retriever().retrieve(question)
routed  = route(question, intents, chunks)

# routed.prompt → send to LLM
# routed.variant → log to MLflow
```

---

## Templates

Every template is a plain Python string with two format slots:

- `{context}` — the assembled chunk context injected by `router.route`
- `{question}` — the user's question

All templates enforce a **document-grounded rule**: the LLM must not use outside knowledge and must explicitly say so when information is missing.

### `lookup.py` — `LOOKUP_PROMPT`

**Variant**: `lookup_v1`

Used when the user wants a specific value that is recorded in the document (e.g. interest rate, applicant name, branch, date).

- Quotes exact values directly from the document.
- Responds with `"Not found in the document."` if the value is absent.
- No explanation unless explicitly asked.

**Example triggers**: *"What is the interest rate?"*, *"Who is the primary applicant?"*, *"What date was the loan sanctioned?"*

---

### `calculate.py` — `CALCULATE_PROMPT`

**Variant**: `calculate_v1`

Used when the user wants arithmetic performed using values in the document (EMI, total repayment, cost of borrowing, etc.).

- Shows step-by-step working.
- States the final answer with the correct unit (₹, %, months).
- Lists any missing values rather than estimating them.

**Example triggers**: *"Calculate the total amount I will pay over 36 months"*, *"What is the total interest cost?"*

---

### `compare.py` — `COMPARE_PROMPT`

**Variant**: `compare_v1`

Used when the user wants two or more document values placed side by side.

- Presents comparisons as a table or bullet list.
- Labels each value by entity (e.g. Applicant vs Co-Applicant).
- Explicitly flags missing values rather than inferring them.

**Example triggers**: *"Compare the applicant and co-applicant income"*, *"Which is higher — the EMI or the monthly income?"*

---

### `explain.py` — `EXPLAIN_PROMPT`

**Variant**: `explain_v1`

Used when the user wants a term, clause, or section explained in plain language.

- Targets 2–4 sentences; expands only when genuinely needed.
- Uses simple language, minimises jargon.
- Responds with `"This is not covered in the provided document."` when the clause is absent.

**Example triggers**: *"Explain what happens if I miss a payment"*, *"What does this clause mean?"*

---

### `summarise.py` — `SUMMARISE_PROMPT`

**Variant**: `summarise_v1`

Used for broad overview questions and as the **fallback intent** when no other intent scores above threshold.

Produces a structured summary with four sections (if information is available):

1. Applicant Details
2. Loan Details (amount, tenure, rate, repayment mode)
3. Financial Summary (income, expenses, assets, liabilities)
4. Key Observations

Each section is kept to 2–3 sentences.

**Example triggers**: *"Give me a summary of this document"*, *"What is this document about?"*, or any query that doesn't clearly match another intent.

---

## Adding a new template

1. Create `app/prompts/templates/<intent_name>.py` with a `<INTENT>_PROMPT` string containing `{context}` and `{question}` slots.
2. Add the new `Intent` enum value in `app/retrieval/intent_classifier.py` — add descriptions and insert it into `INTENT_PRIORITY`.
3. Register the template in `router.py` — add an entry to `_TEMPLATE_MAP` with a versioned variant label (e.g. `"myintent_v1"`).
4. Write tests in `tests/retrieval/test_intent_classifier.py` (classification) and `tests/prompts/` (routing + template rendering).

---

## Versioning

Variant labels (`lookup_v1`, `calculate_v1`, …) are logged to MLflow in Phase 3 for A/B prompt experiments. When you update a template's instructions, bump the version suffix (`v1` → `v2`) so experiment runs stay comparable.
