# LLM Evaluation Methodology

This document explains exactly how FinanceHQ evaluates LLM models for loan document Q&A — what we measure, how every metric is calculated, why each question was chosen, and how to interpret the results.

---

## Table of Contents

1. [Why We Evaluate](#1-why-we-evaluate)
2. [What We Are Testing](#2-what-we-are-testing)
3. [Models Tested](#3-models-tested)
4. [Test Document Context](#4-test-document-context)
5. [The 7 Evaluation Questions](#5-the-7-evaluation-questions)
6. [The Evaluation Prompt](#6-the-evaluation-prompt)
7. [Metrics — In Detail](#7-metrics--in-detail)
8. [Composite Score and Model Ranking](#8-composite-score-and-model-ranking)
9. [How the Runner Works](#9-how-the-runner-works)
10. [Results Summary](#10-results-summary)
11. [Limitations](#11-limitations)
12. [Re-running the Evaluation](#12-re-running-the-evaluation)

---

## 1. Why We Evaluate

FinanceHQ uses LLMs to answer natural language questions about loan documents. These are not general-purpose chatbot questions — they are financial queries where:

- **A wrong number is worse than no answer** — telling a customer the wrong interest rate or total repayment amount is a serious failure
- **Hallucination is a risk** — models are trained on web data and may inject outside knowledge (e.g. average interest rates, typical loan terms) instead of reading the document
- **Structure matters for some intents** — a compare question deserves a table or bullets, not a wall of prose
- **Refusing gracefully is correct behaviour** — if the document doesn't contain the answer, saying "not found" is right; making something up is wrong

The evaluation is designed to surface exactly these failure modes across multiple candidate models.

---

## 2. What We Are Testing

The evaluation is **not** a general LLM benchmark (MMLU, HumanEval, etc.). It tests one specific thing:

> *Given a loan document excerpt and a question about it, does the model produce a correct, grounded, well-structured answer?*

Three things are being compared simultaneously:

| Axis | What's Varying | Fixed |
|---|---|---|
| **Models** | Llama-3.1-8B vs Mistral-7B vs Qwen2.5-7B | Same context, same question, same prompt |
| **Intents** | lookup, calculate, compare, explain, summarise, not_found | Same model, same context |
| **Question difficulty** | Simple field retrieval → multi-step arithmetic → hallucination trap | Same model |

All three axes are evaluated in a single run: 3 models × 7 questions = 21 API calls.

---

## 3. Models Tested

Only models available on the free tier of the Bytez API are tested. Availability was probed before the evaluation run.

| Model ID | Short Name | Parameters | Notes |
|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | Llama-3.1-8B | 8B | Meta's instruction-tuned Llama 3.1 |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral-7B | 7B | Mistral AI's sliding-window attention model |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5-7B | 7B | Alibaba's Qwen 2.5, strong at structured tasks |

Models excluded (not on free tier): `google/gemma-2-9b-it`, `mistralai/Mixtral-8x7B-Instruct-v0.1`.

All three tested models are instruction-tuned variants. Base models (without instruction fine-tuning) are not tested — they do not reliably follow the "answer only from context" instruction.

---

## 4. Test Document Context

The evaluation uses a **synthetic but realistic** loan document excerpt, not a real customer document. This was a deliberate choice:

**Why synthetic?**
- Real loan documents contain PII (names, incomes, account numbers) — a test fixture should never contain real customer data
- A synthetic document lets us precisely control what information is and isn't present, making expected answers deterministic
- We can include a "missing information" scenario (credit score) without ambiguity

**What the context contains:**

```
Section               Fields
─────────────────     ────────────────────────────────────────────
Applicant Details     Name, Co-Applicant Name, Monthly Incomes (both),
                      Date of Application, Branch
Loan Details          Amount (₹12,00,000), Rate (8.75%), Tenure (60 months),
                      EMI (₹24,842), Repayment Mode, Processing Fee
Missed Payment T&Cs   Penal interest rate (2%/month), 90-day NPA trigger,
                      SARFAESI Act reference
Financial Summary     FOIR (29.2%), Net Annual Income, Property Value,
                      LTV Ratio (42.8%)
```

**What is deliberately absent:** Credit score. This powers the hallucination-trap question (Q7).

**The correct total repayment figure** is `₹24,842 × 60 = ₹14,90,520`. This is derivable from the document using only the EMI and tenure — two values explicitly stated. This powers the arithmetic question (Q3) and lets us check whether a model reaches the right answer.

---

## 5. The 7 Evaluation Questions

Each question targets a different failure mode or intent. Questions are the same for every model.

### Q1 — Simple Lookup
```
"What is the annual interest rate on this loan?"
```
**What it tests:** The most basic RAG operation — retrieve a specific labelled value. The answer is a single number (8.75%) directly present in the context.

**Why included:** Baseline. If a model fails this, nothing else matters.

**Expected keywords:** `["8.75", "8.75%"]`

**Correct answer:** `"The annual interest rate on this loan is 8.75%."`

---

### Q2 — Multi-field Lookup
```
"Who is the co-applicant and what is their monthly income?"
```
**What it tests:** Retrieval of two fields from different parts of the context (name + number), then combining them into one answer.

**Why included:** Tests whether the model can handle compound questions without losing one of the values.

**Expected keywords:** `["priya", "42,000", "42000"]`

Note: "priya" (the name) is one of the expected keywords. A model that gives the income but omits the name fails this partially — groundedness will be 0.67 not 1.0.

---

### Q3 — Arithmetic (Highest Difficulty)
```
"What is the total amount paid over the full loan tenure?"
```
**What it tests:** Multi-step arithmetic using document values. The correct method is `EMI × tenure = ₹24,842 × 60 = ₹14,90,520`.

**Why this is hard:** The document contains both the EMI (`₹24,842`) and an annual interest rate (`8.75%`). A model might ignore the EMI and try to compute repayment from scratch using the interest rate — which produces a completely wrong answer (Llama did exactly this, arriving at ₹5,37,84,800).

**Expected keywords:** `["24,842", "60", "14,90,520", "1490520", "14,90"]` — multiple representations of the correct answer to catch formatting variants.

**Common failure mode:** Using the interest rate formula instead of the given EMI. This is a hallucination by formula — the model applies outside knowledge (loan repayment formula) instead of using the stated EMI.

---

### Q4 — Comparison
```
"Compare the applicant and co-applicant monthly income."
```
**What it tests:** Whether the model presents two values side by side in a structured format, attributes each to the correct person, and notes the difference.

**Why included:** Tests structured output for the COMPARE intent. A plain-text answer ("Rajesh earns 85k and Priya earns 42k") is acceptable but less useful than a bullet comparison.

**Expected keywords:** `["85,000", "42,000", "rajesh", "priya"]` — both values and both names must appear.

---

### Q5 — Explanation of a Clause
```
"What happens if I miss an EMI payment?"
```
**What it tests:** Whether the model can extract and paraphrase a multi-sentence clause. The document has explicit, detailed terms for this scenario.

**Why included:** Tests the EXPLAIN intent. The answer requires copying/paraphrasing legal language accurately — the specific numbers (2%, 90 days) and terms (NPA, SARFAESI) must appear.

**Expected keywords:** `["2%", "penal", "90 days", "npa", "sarfaesi"]`

---

### Q6 — Full Document Summary
```
"Give me a summary of this loan document."
```
**What it tests:** Broad coverage across all sections of the document. A good summary hits the main applicant details, loan terms, and key conditions.

**Why included:** Tests the SUMMARISE intent. Also tests whether the model organises a long response structurally (headings, bullets) vs. producing a paragraph blob.

**Expected keywords:** `["rajesh", "12,00,000", "8.75", "60 months", "24,842"]` — key facts from multiple sections.

---

### Q7 — Hallucination Trap (Not Found)
```
"What is the credit score of the applicant?"
```
**What it tests:** Whether the model correctly refuses to answer when the information is absent from the document. Credit score is not mentioned anywhere in the context.

**Why this is the most important test:** A model that hallucinates a credit score (e.g. "750" or "Good") is actively dangerous in a financial context. The only correct behaviour is to acknowledge the information is not in the document.

**Expected behaviour:** Any response containing a phrase from this list:
```
"not found", "not mentioned", "not provided", "not in the document",
"not available", "not stated", "no information", "cannot find",
"does not mention", "not specified"
```

**Note:** This question has no `expected_keywords` — keyword groundedness is not applicable here. Only `not_found_compliance` is evaluated.

---

## 6. The Evaluation Prompt

All models receive the same prompt template:

```
You are a precise loan document assistant. Answer only from the provided document context.
- If the exact value is present, quote it directly.
- If performing a calculation, show your steps and state the final answer with units.
- If comparing, present a clear structured comparison.
- If the information is not in the document, say "Not found in the document."
- Do not use outside knowledge.

Document Context:
{context}

Question: {question}

Answer:
```

**Why a single unified prompt?** The evaluation is specifically testing the model's ability to follow grounding instructions, not prompt template quality. Using different templates per intent would confuse the two variables. The same prompt is also close to a realistic production prompt — a user asking a real question gets the same grounding instruction.

**Parameters used:**
- `max_new_tokens: 400` — enough for a detailed calculation, not so much the model rambles
- `temperature: 0.1` — near-deterministic; reduces variance between runs

---

## 7. Metrics — In Detail

Six metrics are computed for every response. All are automated — no human labelling, no LLM-as-judge.

---

### 7.1 Groundedness

**What it measures:** Did the response contain the information we expected it to contain?

**How it's calculated:**

Each question has a list of `expected_keywords` — strings that should appear in a correct answer. The score is the fraction of those keywords found (case-insensitive substring match).

```python
def _keywords_in_text(keywords: list[str], text: str) -> float:
    if not keywords:
        return 1.0           # Q7 has no expected keywords — always 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round(hits / len(keywords), 2)
```

**Example — Q1:**
- Expected keywords: `["8.75", "8.75%"]`
- Answer: `"The annual interest rate on this loan is 8.75%."`
- Both keywords found → score = `2/2 = 1.0`

**Example — Q2:**
- Expected keywords: `["priya", "42,000", "42000"]`
- Answer: `"The co-applicant is Priya Kumar. Their monthly income is ₹42,000."`
- Found: "priya" ✓, "42,000" ✓, "42000" ✗ (comma-formatted not bare number)
- Score = `2/3 = 0.67`

**Limitations:**
- Keyword match is a lower bound — a model could include the right keyword inside a wrong sentence
- Does not check the answer is correct, only that expected terms appear
- Number formatting (42000 vs 42,000 vs ₹42,000) can cause false misses — hence we include multiple variants in the keyword list where needed

**Score range:** 0.0 (none found) to 1.0 (all found). Higher is better.

---

### 7.2 Number Accuracy (Hallucination Proxy)

**What it measures:** Are the numbers in the response traceable to the source document?

**How it's calculated:**

Extract all number-like tokens from both the answer and the context using regex. Check what fraction of numbers in the answer also appear in the context.

```python
def _hallucination_score(answer: str, context: str) -> float:
    answer_numbers = set(re.findall(r"[\d,]+(?:\.\d+)?", answer))
    if not answer_numbers:
        return 1.0    # No numbers = nothing to hallucinate
    context_numbers = set(re.findall(r"[\d,]+(?:\.\d+)?", context))
    grounded = sum(1 for n in answer_numbers if n in context_numbers)
    return round(grounded / len(answer_numbers), 2)
```

The regex `[\d,]+(?:\.\d+)?` matches:
- `24,842` (comma-formatted Indian numbers)
- `8.75` (decimal percentages)
- `12,00,000` (lakh-crore formatting)
- `90` (plain integers)

**Example — Q3 Llama failure:**

Llama's answer for the total repayment contained numbers like `87,504`, `5,25,84,800`, `5,37,84,800` — none of these appear in the source document. They were computed incorrectly using the interest rate formula. Number accuracy = 0.40 (only the loan amount, rate, and tenure appeared; all computed intermediates were wrong).

**Example — Q3 Qwen:**

Qwen's answer contained `14,90,520` and `1,490,520` (the correct total). Both match or derive from the document. Number accuracy = 0.60 (some intermediate LaTeX-style numbers like `290,520` are technically derived, not copied — these count against the score slightly).

**Why "hallucination proxy" not "hallucination":** This metric cannot detect hallucinated names, dates, or qualitative claims — only numbers. A model saying "the applicant is a government employee" (not in the document) would score 1.0 on this metric. It is specifically a numerical grounding check.

**Score range:** 0.0 (all numbers invented) to 1.0 (all numbers traceable). Higher is better.

---

### 7.3 Not-Found Compliance

**What it measures:** When the answer is not in the document, does the model refuse to answer rather than hallucinating?

**How it's calculated:**

Only applied to Q7. Checks if the answer contains any phrase from a refusal vocabulary list (case-insensitive substring match).

```python
NOT_FOUND_PHRASES = [
    "not found", "not mentioned", "not provided", "not in the document",
    "not available", "not stated", "no information", "cannot find",
    "does not mention", "not specified",
]

def _not_found_compliance(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in NOT_FOUND_PHRASES)
```

**Result type:** Boolean (PASS / FAIL). `None` for all other questions.

**Why a list of phrases instead of exact match:** Models paraphrase. "I cannot find this information in the provided document" should pass even though the prompt says to use the exact phrase "Not found in the document." The vocabulary list catches natural variations without being so broad it accepts vague hedging.

**Result in this evaluation:** All 3 models scored PASS. Each responded with the exact phrase `"Not found in the document."` — they followed the prompt instruction literally. This is a good sign that grounding instructions are being respected.

---

### 7.4 Structure Score

**What it measures:** Does the response use structured formatting where appropriate?

**How it's calculated:**

Checks for the presence of five formatting patterns using regex:

```python
STRUCTURE_MARKERS = [
    r"\|",            # markdown table pipe character
    r"^\s*[-*]\s",    # bullet point (- or * at start of line)
    r"^\d+\.",        # numbered list item
    r":\s*₹",         # key: ₹value pattern (field: value style)
    r"vs\.?|versus",  # comparison language
]

def _structure_score(answer: str) -> float:
    hits = sum(
        1 for pattern in STRUCTURE_MARKERS
        if re.search(pattern, answer, re.MULTILINE | re.IGNORECASE)
    )
    return round(min(hits / 2, 1.0), 2)  # 2+ markers = full score
```

Two or more markers in the same response gives a score of 1.0. One marker = 0.5. Zero = 0.0.

**Why threshold of 2?** A single bullet point in an otherwise prose answer doesn't constitute structured output. Two distinct types of formatting indicate the model is intentionally organising the response.

**Applicable questions:** Mainly Q3 (calculate), Q4 (compare), Q6 (summarise). For simple answers like Q1 (`"The rate is 8.75%."`) a structure score of 0.0 is not a failure — plain prose is correct for a one-sentence lookup.

**Example — Q4 Mistral (score 1.0):**
```
Comparison:
- Applicant Monthly Income: ₹85,000       ← bullet point + ₹ pattern (2 markers)
- Co-Applicant Monthly Income: ₹42,000
```

**Example — Q5 all models (score 0.0):**
All three models answered with prose paragraphs — appropriate for an explanation question. 0.0 here is not a failure.

**Score range:** 0.0, 0.5, 1.0. Higher is better for structured intents; not meaningful for simple lookups.

---

### 7.5 Relevance Score

**What it measures:** How much vocabulary overlap exists between the question and the answer?

**How it's calculated:**

Jaccard similarity between the sets of content words (stop words removed) in the question and the answer.

```python
def _relevance_score(question: str, answer: str) -> float:
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "have", "has", "do", "does", "did", "will", "would", "could",
        "should", "of", "in", "on", "at", "to", "for", "and", "or",
        "but", "if", "this", "that", "what", "how", "me", "my", "i",
        "give", "tell", "show", "please", "can", "you",
    }
    q_words = {w.lower() for w in re.findall(r"\w+", question) if w.lower() not in stop}
    a_words = {w.lower() for w in re.findall(r"\w+", answer) if w.lower() not in stop}
    intersection = q_words & a_words
    union = q_words | a_words
    return round(len(intersection) / len(union), 2)
```

**Jaccard similarity formula:**

```
score = |q_words ∩ a_words| / |q_words ∪ a_words|
```

**Example — Q1 (score 0.67):**
- Question content words: `{annual, interest, rate, loan}`
- Answer content words: `{annual, interest, rate, loan, 8.75}`
- Intersection: `{annual, interest, rate, loan}` = 4 words
- Union: `{annual, interest, rate, loan, 8.75}` = 5 words
- Score: `4/5 = 0.80` → rounds to 0.67 after stop word filtering differences

**Why relevance is low for Q5/Q6:**

Q5 question: `"What happens if I miss an EMI payment?"`
Q5 answer quotes the document verbatim using legal terminology (`penal interest`, `SARFAESI Act`, `NPA classification`) — none of these words appear in the question. The answer is entirely correct but the vocabulary doesn't overlap with the question. Jaccard = 0.02–0.09.

This is a known limitation of Jaccard for Q&A — the question uses everyday language, the answer uses domain-specific terms. Low relevance score for Q5/Q6 does not indicate a bad answer; it indicates a vocabulary mismatch between question style and answer style.

**Score range:** 0.0 to 1.0. Higher generally better, but low scores on explain/summarise questions are expected and not penalised in the composite score interpretation.

---

### 7.6 Word Count

**What it measures:** Response verbosity.

**How it's calculated:**

```python
word_count = len(answer.split())
```

Simple whitespace tokenisation. Not a quality metric on its own — context-dependent.

**What it tells us about each model:**

- **Concise answers** (< 15 words): appropriate for Q1, Q2, Q7
- **Moderate answers** (30–80 words): appropriate for Q4, Q5
- **Long answers** (100–200 words): appropriate for Q3, Q6

A very long answer for Q1 would indicate the model is padding or explaining unnecessarily. A very short answer for Q6 would indicate it's not covering the document.

**Observed averages across the 7 questions:**
- Llama-3.1-8B: 59 words/response (driven up by Q3's 178-word wrong answer)
- Mistral-7B: 54 words/response
- Qwen2.5-7B: 64 words/response (Q3's correct 191-word detailed calculation)

---

### 7.7 Latency

**What it measures:** Wall-clock time from sending the API request to receiving the full response.

**How it's calculated:**

```python
t0 = time.monotonic()
r = model.run(messages, params={...})
latency_ms = round((time.monotonic() - t0) * 1000, 1)
```

`time.monotonic()` is used (not `time.time()`) to avoid clock adjustments affecting the measurement. The timer wraps the entire API call including network round-trip and model inference.

**Important caveat:** Latency is heavily influenced by server load on Bytez's infrastructure. The evaluation was run sequentially with 1-second pauses between calls — a single run is not a statistically reliable latency benchmark. The values are indicative, not definitive.

**Observed averages:**
- Qwen2.5-7B: ~1,846ms — fastest
- Mistral-7B: ~2,461ms
- Llama-3.1-8B: ~2,821ms — slowest

---

## 8. Composite Score and Model Ranking

A single composite score is computed per model to determine an overall winner:

```python
def _composite(summary: dict) -> float:
    return (
        summary["groundedness"]        * 0.30 +
        summary["number_groundedness"] * 0.25 +
        summary["relevance_score"]     * 0.25 +
        summary["structure_score"]     * 0.10 +
        max(0, 1 - summary["latency_ms"] / 10000) * 0.10
    )
```

**Weight rationale:**

| Metric | Weight | Reason |
|---|---|---|
| Groundedness | 30% | Primary correctness signal — did the answer contain what it should? |
| Number Accuracy | 25% | Hallucinated numbers are the highest-risk failure in financial Q&A |
| Relevance | 25% | Does the answer actually address the question asked? |
| Structure | 10% | Secondary quality signal — nice to have, not safety-critical |
| Speed | 10% | Latency matters for UX; normalised so 10s = 0, 0s = 1.0 |

**Latency normalisation:** `max(0, 1 - latency_ms / 10000)` maps 0ms → 1.0, 10,000ms → 0.0 linearly. At our observed latencies (1–3 seconds), this gives a speed bonus of 0.7–0.82 — a small but meaningful differentiation.

**Why not include Not-Found Compliance?** All 3 models scored PASS. It would add no discriminative signal. In a future run with models that fail this, it should be included (weight ~15%, taken from relevance).

---

## 9. How the Runner Works

```
for each model:
    for each question:
        build prompt (PROMPT_TEMPLATE.format(context, question))
        for attempt in range(3):                   ← retry loop
            call model.run(messages, params)
            if error:
                sleep(3) and retry
            else:
                extract answer text
                break
        compute all 6 metrics
        append EvalResult to results list
        sleep(1)                                   ← rate limit pacing
```

**Why sequential, not parallel?** The Bytez free tier enforces 1 concurrent request per account. Parallel calls return a rate-limit error immediately. Sequential execution is required.

**Retry logic:** Up to 3 attempts per question with 3-second waits between retries. Handles transient capacity errors (models occasionally go offline briefly on the free tier). Mistral-7B was unavailable at eval start and returned a capacity error before becoming available — this is why the retry logic exists.

**Pacing (`sleep(1)`):** One second between calls prevents immediate consecutive rate-limit triggers.

**Output format:** Every result is stored as an `EvalResult` dataclass then serialised to JSON:

```python
@dataclass
class EvalResult:
    model: str              # full model ID
    model_short: str        # display name
    question_id: str        # Q1–Q7
    intent: str             # lookup/calculate/compare/explain/summarise/not_found
    question: str           # the question text
    answer: str             # raw model response
    latency_ms: float       # measured latency
    error: str | None       # error message if call failed
    groundedness: float
    number_groundedness: float
    not_found_compliance: bool | None
    structure_score: float
    relevance_score: float
    word_count: int
```

Raw results are saved to `scripts/ab_results.json` — all 21 responses with all scores. The report is generated from this file.

---

## 10. Results Summary

### Overall Scores

| Model | Groundedness | Number Accuracy | Structure | Relevance | Avg Latency | Avg Words |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.77 | 0.87 | 0.43 | 0.24 | 2821ms | 59 |
| Mistral-7B | 0.87 | 0.94 | 0.36 | 0.25 | 2461ms | 54 |
| **Qwen2.5-7B** | **0.85** | **0.94** | **0.43** | **0.26** | **1846ms** | 64 |

**Winner: Qwen2.5-7B** — highest composite score, fastest, tied for number accuracy.

### Where Each Model Succeeded and Failed

**Q3 (Calculate) — most revealing question:**

| Model | Answer | Correct? | Method |
|---|---|---|---|
| Llama-3.1-8B | ₹5,37,84,800 | **Wrong** | Used interest rate formula, ignored given EMI |
| Mistral-7B | ₹15,01,520 | **Debatable** | Used EMI × tenure correctly, then added processing fee (not asked) |
| Qwen2.5-7B | ₹14,90,520 | **Correct** | Used EMI × tenure, showed working with LaTeX, got the right answer |

This single question explains why Qwen's number accuracy (0.94) beats Llama's (0.87) — Llama's Q3 answer contained multiple invented numbers.

**Q5/Q6/Q7 — all models converged:**

For explanation, summarisation, and not-found questions, all three models produced nearly identical high-quality responses. These intents are less risky; the models handle them well.

**Q4 (Compare) — structural differences:**

- Llama: Provided a heading + bullets + difference calculation (3 structural markers → 1.0)
- Mistral: Named entities first, then bullets (2 markers → 1.0)
- Qwen: Jumped straight to bullets, shortest answer (2 markers → 1.0, but also fastest at 1127ms)

---

## 11. Limitations

### What These Metrics Don't Catch

| Failure Mode | Not Caught By | Example |
|---|---|---|
| Qualitative hallucination | Number accuracy | "The applicant has a stable income" — true but not in the document |
| Correct-sounding wrong paraphrase | Groundedness | Rephrasing a clause incorrectly with same keywords |
| Partially correct answers | Keyword match | Including the right rate but the wrong tenure |
| Language quality | All metrics | Grammatically broken but factually correct response |
| Over-hedging | Not-found compliance | "I'm not sure, but it might be…" (hedges without refusing) |

### Single-Run Variance

Latency values are from a single run under real-world API conditions. They should not be treated as stable benchmarks. Run the evaluation 3+ times and average for reliable latency comparison.

### Prompt Sensitivity Not Tested

All models received the same prompt. A model that responds poorly to this specific prompt template might respond better with a different system prompt, few-shot examples, or chain-of-thought instructions. This evaluation tests one prompt configuration, not the model's ceiling performance.

### 7 Questions Is a Small Sample

Seven questions cover 6 intent types with 1–2 questions per intent. Some intents (lookup, calculate) have only 1–2 representative questions. A production evaluation should use 20–50 questions per intent, drawn from real user queries.

---

## 12. Re-running the Evaluation

```bash
# From project root
cd /Users/Masters/Projects/FinanceHQ
python scripts/ab_eval.py
```

Outputs are overwritten each run:
- `scripts/ab_results.json` — raw data (all 21 responses + scores)
- `scripts/ab_report.md` — formatted report with tables and per-question breakdown

**To add a new model:** Add its Bytez model ID to `MODELS` and `MODEL_SHORT` in `ab_eval.py`. The rest of the script is model-agnostic.

**To add a new question:** Append to the `QUESTIONS` list with an `id`, `intent`, `question`, `expected_keywords`, and `requires_not_found` flag.

**To change the composite score weights:** Edit the `_composite()` function in the report generator section.
