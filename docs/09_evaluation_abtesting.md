# Evaluation and A/B Testing

## What This Section Covers

- How individual responses are evaluated for quality (using Bytez as LLM-judge)
- How prompt variants are A/B tested to find the best-performing template
- How results accumulate in MLflow for analysis

---

## Response Evaluation

### LLM-as-Judge Pattern

After every query, a separate Bytez API call evaluates the quality of the answer. This is the "LLM-as-judge" pattern — using a language model to score another language model's output.

The evaluator receives:
- The original question
- The retrieved context chunks
- The generated answer

It scores on two dimensions:

### Faithfulness Score (0.0 – 1.0)

"Is every claim in the answer supported by the retrieved context?"

```
Evaluator prompt:

You are evaluating the faithfulness of an AI-generated answer.

RETRIEVED CONTEXT:
{context}

QUESTION: {question}

GENERATED ANSWER: {answer}

Rate the faithfulness of the answer on a scale of 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the context
- 0.5: Most claims are supported, but some are inferred or not directly stated
- 0.0: The answer contains claims not found in the context (hallucination)

Return only a JSON object: {"faithfulness": 0.0}
```

### Relevance Score (0.0 – 1.0)

"Does the answer actually address what was asked?"

```
Evaluator prompt:

You are evaluating answer relevance.

QUESTION: {question}
GENERATED ANSWER: {answer}

Rate how well the answer addresses the question (0.0 to 1.0):
- 1.0: Directly and completely answers the question
- 0.5: Partially answers, or answers a related but different question
- 0.0: Does not answer the question at all

Return only a JSON object: {"relevance": 0.0}
```

### Evaluator Model

Uses a smaller/cheaper Bytez model for evaluation (e.g., `mistralai/Mistral-7B-Instruct` vs `Mixtral` for inference). Evaluation calls are cheaper than inference calls.

### Composite Score

```python
composite_score = 0.6 * faithfulness + 0.4 * relevance
```

Faithfulness is weighted higher because hallucination is the primary risk in RAG systems — relevance failures are more visible (user notices) but hallucinations are more dangerous (user doesn't notice).

---

## A/B Testing Framework

### Goal

Determine which prompt variant (A or B) produces higher quality answers, as measured by the composite Bytez evaluation score.

### What Is Varied

For each intent type, two prompt variants exist (see `07_prompt_engineering.md`). The A/B test runs across all queries to accumulate statistical evidence.

Possible axes of variation per experiment:

| Parameter | Variant A | Variant B |
|---|---|---|
| Prompt verbosity | Detailed instructions | Minimal instructions |
| Context format | Labelled `[Source N]` | Raw concatenated text |
| Answer format instruction | "Use bullet points" | "Be concise" |
| k (top chunks) | 5 | 3 |
| Temperature | 0.1 | 0.3 |

**One parameter changes at a time.** Do not A/B test multiple parameters simultaneously — you cannot attribute the score difference to a specific change.

### Variant Assignment

```python
import random

def select_variant(session_id: str, intent: str) -> str:
    """
    50/50 random assignment. Session-stable within a session
    (same user, same session always gets the same variant).
    """
    seed = hash(f"{session_id}:{intent}") % 100
    return "A" if seed < 50 else "B"
```

Session-stable assignment means a single user won't see inconsistent behavior within one session. Across sessions, the distribution converges to 50/50.

### Experiment Configuration

Managed via a config file (not hardcoded), so new experiments can be started without code changes:

```json
// experiment_config.json (stored in S3 or env var)
{
  "active_experiment": "exp_003_context_format",
  "variants": {
    "A": {
      "description": "Labelled source format",
      "k": 5,
      "context_format": "labelled",
      "temperature": 0.1
    },
    "B": {
      "description": "Raw concatenated format",
      "k": 5,
      "context_format": "raw",
      "temperature": 0.1
    }
  },
  "started_at": "2025-01-20",
  "target_samples_per_variant": 50
}
```

### Stopping Rule

An experiment is concluded when:
1. Each variant has ≥ 50 evaluated queries, AND
2. The difference in mean composite scores is ≥ 0.05 (practical significance threshold)

If after 200 queries there is no significant difference, both variants are declared equivalent and the simpler one (A) is adopted as default.

---

## MLflow Integration for A/B Results

Every query creates one MLflow run under the experiment `loandoc_rag_ab_testing`:

```python
with mlflow.start_run(experiment_id=EXPERIMENT_ID):
    # Parameters (what was used)
    mlflow.log_param("variant", "A")
    mlflow.log_param("intent", "finance")
    mlflow.log_param("k", 5)
    mlflow.log_param("model", "mistralai/Mistral-7B-Instruct-v0.3")
    mlflow.log_param("temperature", 0.1)
    mlflow.log_param("context_format", "labelled")

    # Metrics (what happened)
    mlflow.log_metric("faithfulness_score", 0.87)
    mlflow.log_metric("relevance_score", 0.91)
    mlflow.log_metric("composite_score", 0.886)
    mlflow.log_metric("retrieval_latency_ms", 45)
    mlflow.log_metric("inference_latency_ms", 1240)
    mlflow.log_metric("eval_latency_ms", 890)
    mlflow.log_metric("input_tokens", 847)
    mlflow.log_metric("output_tokens", 124)

    # Artifacts (the actual content)
    mlflow.log_text(question, "question.txt")
    mlflow.log_text(prompt, "prompt.txt")
    mlflow.log_text(answer, "answer.txt")
    mlflow.log_dict({"chunks": chunks_used}, "retrieved_chunks.json")
```

### Querying Results in MLflow

To compare variant A vs B after accumulating data:
```python
import mlflow

runs = mlflow.search_runs(
    experiment_ids=[EXPERIMENT_ID],
    filter_string="params.intent = 'finance'",
)

variant_a = runs[runs["params.variant"] == "A"]["metrics.composite_score"]
variant_b = runs[runs["params.variant"] == "B"]["metrics.composite_score"]

print(f"Variant A mean: {variant_a.mean():.3f} (n={len(variant_a)})")
print(f"Variant B mean: {variant_b.mean():.3f} (n={len(variant_b)})")
```

This analysis can be run in the MLflow UI (built-in charts) or as a Jupyter notebook.

---

## Evaluation Cadence

| Event | Evaluation triggered? |
|---|---|
| Every `/query` call | Yes — always evaluate |
| Cached query hit | No — skip eval (use cached score) |
| Pipeline failure | No — no answer to evaluate |
| Streaming response | Yes — evaluate after stream completes |

Evaluation adds ~1–2 seconds of latency (async, non-blocking for the user). The response is returned to the user immediately; evaluation and MLflow logging happen in a background task.
