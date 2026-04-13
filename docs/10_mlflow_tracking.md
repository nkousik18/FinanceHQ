# MLflow Experiment Tracking

## What MLflow Tracks in This Project

MLflow is not just for model training here. It tracks every query as an experiment — making the prompt engineering and A/B testing process fully reproducible and analyzable.

This is the key MLOps differentiator of this project: **every production query is a logged experiment.**

## MLflow Setup

### Tracking Server

A self-hosted MLflow server deployed as a separate Render service.

```bash
mlflow server \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:5432/${POSTGRES_DB} \
    --default-artifact-root s3://loandoc-bucket/mlflow/artifacts/ \
    --host 0.0.0.0 \
    --port 5000
```

| Component | Where |
|---|---|
| Tracking DB | Render Postgres (free 1GB) |
| Artifact store | S3 `mlflow/artifacts/` |
| Server | Render web service (free tier) |
| UI | `https://loandoc-mlflow.onrender.com` |

**Important:** Render free tier services sleep after 15 minutes of inactivity. For the MLflow server, use `UptimeRobot` (free) to ping `/health` every 10 minutes and keep it awake.

### Client Configuration in FastAPI

```python
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
# Example: "https://loandoc-mlflow.onrender.com"

mlflow.set_experiment("loandoc_rag_ab_testing")
```

### AWS Credentials for Artifact Store

MLflow server needs S3 access to store artifacts. Set on the Render MLflow service:
```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
MLFLOW_S3_BUCKET=loandoc-bucket
```

---

## Experiment Structure

### Experiments

| Experiment Name | Purpose |
|---|---|
| `loandoc_rag_ab_testing` | Main experiment — all production queries |
| `loandoc_pipeline_runs` | Pipeline execution runs (extraction, chunking stats) |
| `loandoc_prompt_dev` | Development-time prompt testing (not production queries) |

### Run Structure

Each query → one MLflow run in `loandoc_rag_ab_testing`:

```
run/
├── params/
│   ├── variant          # "A" or "B"
│   ├── intent           # "finance", "summary", etc.
│   ├── k                # chunks retrieved
│   ├── model            # Bytez model name
│   ├── temperature
│   ├── context_format
│   └── session_id
│
├── metrics/
│   ├── faithfulness_score
│   ├── relevance_score
│   ├── composite_score
│   ├── retrieval_latency_ms
│   ├── inference_latency_ms
│   ├── eval_latency_ms
│   ├── total_latency_ms
│   ├── input_tokens
│   ├── output_tokens
│   └── estimated_cost_usd
│
└── artifacts/
    ├── question.txt
    ├── prompt.txt
    ├── answer.txt
    └── retrieved_chunks.json
```

### Pipeline Run Structure

Each document processing pipeline → one MLflow run in `loandoc_pipeline_runs`:

```
run/
├── params/
│   ├── session_id
│   ├── user_id
│   ├── original_filename
│   ├── chunk_size
│   ├── chunk_overlap
│   └── embedding_model
│
├── metrics/
│   ├── page_count
│   ├── char_count_raw
│   ├── char_count_cleaned
│   ├── chunk_count
│   ├── extraction_latency_ms
│   ├── cleaning_latency_ms
│   ├── chunking_latency_ms
│   └── embedding_latency_ms
│
└── artifacts/
    └── validation_report.json
```

---

## MLflow Logging Code Pattern

All MLflow logging is isolated in a single module `app/tracking.py`:

```python
# app/tracking.py

import mlflow
import asyncio
from functools import wraps

class QueryTracker:

    @staticmethod
    async def log_query_run(
        question: str,
        prompt: str,
        answer: str,
        chunks: list[dict],
        params: dict,
        metrics: dict,
    ):
        """
        Non-blocking: runs in background task, does not block the API response.
        """
        def _log():
            with mlflow.start_run(experiment_id=AB_EXPERIMENT_ID):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_text(question, "question.txt")
                mlflow.log_text(prompt, "prompt.txt")
                mlflow.log_text(answer, "answer.txt")
                mlflow.log_dict({"chunks": chunks}, "retrieved_chunks.json")

        # Run in thread pool to avoid blocking the async event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _log)
```

Called from the FastAPI endpoint as a background task:
```python
background_tasks.add_task(
    QueryTracker.log_query_run,
    question=question,
    prompt=prompt,
    answer=answer,
    chunks=chunks,
    params=params,
    metrics=metrics,
)
```

This ensures MLflow logging never adds latency to the user-facing response.

---

## Prompt Versioning

Prompt templates are versioned via MLflow Model Registry (used as a config store, not a model store):

```python
# Register a new prompt version
mlflow.log_artifact("prompts/finance_v3.txt", artifact_path="prompt_templates")
```

Each experiment run references its prompt template version:
```python
mlflow.log_param("prompt_template_version", "finance_v3")
```

This creates a complete audit trail: given any run in MLflow, you can retrieve the exact prompt that produced that answer.

---

## Reading Results

### Via MLflow UI

`https://loandoc-mlflow.onrender.com` → Experiments → `loandoc_rag_ab_testing`

Built-in charts: compare metric distributions across variants, filter by intent, plot score trends over time.

### Via Python (for analysis)

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("https://loandoc-mlflow.onrender.com")

runs_df = mlflow.search_runs(
    experiment_names=["loandoc_rag_ab_testing"],
    output_format="pandas"
)

# Pivot by variant and intent
summary = runs_df.groupby(["params.variant", "params.intent"]).agg({
    "metrics.composite_score": ["mean", "std", "count"],
    "metrics.total_latency_ms": "mean",
    "metrics.estimated_cost_usd": "sum",
}).round(3)

print(summary)
```

### Cost Tracking

```python
total_cost = runs_df["metrics.estimated_cost_usd"].sum()
print(f"Total Bytez API spend: ${total_cost:.4f}")
```

This gives real-time cost visibility — a direct demonstration of cost optimization awareness.
