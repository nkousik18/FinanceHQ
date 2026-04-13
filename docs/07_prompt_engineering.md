# Prompt Engineering

## Overview

The prompt engineering layer has three jobs:
1. Detect the *intent* of the user's question
2. Select the appropriate *prompt template* (with A/B variant logic)
3. Assemble the final prompt within token budget

## Intent Detection

Rule-based classification — no ML, no API calls, deterministic.

### Intent Categories

| Intent | Trigger Examples | Prompt Behavior |
|---|---|---|
| `finance` | "interest rate", "EMI", "monthly payment", "principal" | Emphasize numerical accuracy, extract figures |
| `summary` | "summarize", "key points", "overview", "what is this document about" | Ask for structured bullet-point summary |
| `explanation` | "explain", "what does X mean", "define", "clarify" | Explain in plain English, avoid jargon |
| `retrieval` | "who", "what", "when", "where", "find", "list" | Direct factual answer with citation |
| `comparison` | "difference between", "vs", "compare" | Side-by-side structure |

### Detection Logic

```python
def detect_intent(question: str) -> str:
    q = question.lower().strip()

    # Hard rules (checked first, highest priority)
    if any(q.startswith(w) for w in ["summarize", "give me a summary", "overview"]):
        return "summary"
    if "explain" in q or "what does" in q or "define" in q:
        return "explanation"
    if any(w in q for w in ["emi", "interest rate", "monthly payment", "principal amount", "repayment schedule"]):
        return "finance"
    if "difference" in q or " vs " in q or "compare" in q:
        return "comparison"

    # Default fallback
    return "retrieval"
```

No fuzzy matching. Hard rules keep behavior predictable and debuggable. If a question matches multiple rules, the first match wins (order above is the priority order).

---

## Prompt Templates

### System Instruction (Shared Across All Intents)

```
You are a precise assistant that answers questions about loan documents.
Answer ONLY based on the provided document excerpts.
If the information is not in the excerpts, say so clearly.
Do not use knowledge from outside the document.
Be concise and direct.
```

This instruction is critical for preventing hallucination. The model must be told explicitly not to use its training knowledge.

### Template: `finance`

```
{system_instruction}

Document excerpts:
{context}

Question: {question}

Instructions:
- Extract the exact numerical value(s) requested
- Include units (%, $, years, months)
- If a calculation is needed, show the calculation
- Quote the relevant clause if helpful
- If the value is not in the document, say "Not specified in this document"

Answer:
```

### Template: `summary`

```
{system_instruction}

Document excerpts:
{context}

Question: {question}

Instructions:
- Provide a structured summary with clear sections
- Focus on: loan amount, interest rate, repayment terms, key obligations, important conditions
- Use bullet points for readability
- 200-300 words maximum

Answer:
```

### Template: `explanation`

```
{system_instruction}

Document excerpts:
{context}

Question: {question}

Instructions:
- Explain in plain English as if the reader has no legal/financial background
- Define any technical terms used
- Reference where in the document this is described (e.g., "According to Section 4")

Answer:
```

### Template: `retrieval`

```
{system_instruction}

Document excerpts:
{context}

Question: {question}

Answer directly and concisely. If the answer is not in the document, state that clearly.

Answer:
```

### Template: `comparison`

```
{system_instruction}

Document excerpts:
{context}

Question: {question}

Instructions:
- Present as a clear comparison
- Use a table format if more than 2 items are compared
- Highlight the key differences

Answer:
```

---

## A/B Variant System

Each template has **two variants** — A (conservative/structured) and B (direct/minimal). These are tested to determine which produces higher evaluation scores.

See `09_evaluation_abtesting.md` for the full A/B testing framework.

### Variant Example: `finance`

**Variant A (structured):**
```
{system_instruction}

Document excerpts:
{context}

Question: {question}

Instructions:
- Extract the exact numerical value(s) requested
- Include units (%, $, years, months)
- Quote the relevant clause
- If not found, say "Not specified in this document"

Answer:
```

**Variant B (minimal):**
```
{system_instruction}

Based on this document:
{context}

{question}

Answer with the specific number or value. Be brief.
```

The A/B system randomly assigns a variant per query (with configurable ratio), logs both the variant and the Bytez evaluation score to MLflow, and gradually builds evidence for which variant performs better.

---

## Token Budget Management

Every assembled prompt is checked against the Bytez API token limit before sending:

```python
MAX_PROMPT_TOKENS = 3000   # leave room for response
MAX_RESPONSE_TOKENS = 512

def assemble_prompt(template: str, context: str, question: str) -> str:
    prompt = template.format(
        system_instruction=SYSTEM_INSTRUCTION,
        context=context,
        question=question,
    )

    tokens = count_tokens(prompt)

    if tokens > MAX_PROMPT_TOKENS:
        # Trim context until prompt fits
        # Remove chunks from the end (lowest-ranked first)
        context = trim_context(context, MAX_PROMPT_TOKENS - count_tokens(
            template.format(system_instruction=SYSTEM_INSTRUCTION, context="", question=question)
        ))
        prompt = template.format(
            system_instruction=SYSTEM_INSTRUCTION,
            context=context,
            question=question,
        )

    return prompt
```

This prevents unexpected costs from very long documents.

### Token Budget Log

Every prompt assembly logs to MLflow:
- `prompt_tokens`: input token count
- `intent`: detected intent
- `variant`: A or B
- `context_chars`: chars of context included
- `chunks_included`: number of chunks that fit
