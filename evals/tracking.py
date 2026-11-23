"""
Local MLflow for the evals (scope v2 §9.2): SQLite backend + local artifact folder under data/mlflow/
(gitignored). Experiments: loandoc_ingestion_evals (S0), loandoc_retrieval_evals (S1–S4),
loandoc_load_tests (S5).

    venv/bin/mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db     # browse

Metric names: MLflow allows letters, digits and _ - . : / space, so "recall@budget" is logged as
"recall_at_budget" and a breakdown group "type=lookup" as "type.lookup".
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data" / "mlflow"
RETRIEVAL_EXPERIMENT = "loandoc_retrieval_evals"
INGESTION_EXPERIMENT = "loandoc_ingestion_evals"
_BAD = re.compile(r"[^\w\-. :/]")
MAX_PARAM_LEN = 6000                   # MLflow 2.x param value limit (6,000 chars)


def use_experiment(name: str, store: Path = STORE):
    """Point MLflow at the local store and select (creating if needed) an experiment."""
    import mlflow
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{store / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(store / "artifacts").as_uri())
    mlflow.set_experiment(name)
    return mlflow


def metric_name(*parts: str) -> str:
    s = ".".join(p.replace("@", "_at_").replace("=", ".") for p in parts if p)
    return _BAD.sub("_", s)


def flat_params(d: dict, prefix: str = "") -> dict[str, str]:
    """Nested config -> flat MLflow params ("run.chunker_params.size": "256")."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flat_params(v, key + "."))
        else:
            out[key] = str(v)[:MAX_PARAM_LEN]
    return out


def retrieval_metrics(summary: dict) -> dict[str, float]:
    """Breakdown means (every group), n per group for the decision metric, latency p50/p95 (ms)."""
    out: dict[str, float] = {}
    for group, ms in summary["breakdown"].items():
        g = "" if group == "overall" else group
        for k, v in ms.items():
            out[metric_name(g, k)] = v["mean"]
        if "recall@budget" in ms:
            out[metric_name(g, "n_answerable")] = ms["recall@budget"]["n"]
    for comp, p in summary["latency_ms"].items():
        for q, v in p.items():
            if v is not None:
                out[metric_name("latency", comp, f"{q}_ms")] = v
    out["unlocated_spans"] = summary["unlocated_spans"]
    out["questions"] = summary["questions"]
    out["run_seconds"] = summary["seconds"]
    for k, v in (summary.get("retriever_info") or {}).items():
        out[metric_name("retriever", k)] = v
    return out


def log_retrieval_run(summary: dict, run_dir: Path, store: Path = STORE) -> str:
    """One MLflow run per configuration (§9.2). Returns the MLflow run id."""
    mlflow = use_experiment(RETRIEVAL_EXPERIMENT, store)
    rc = summary["run"]
    smoke = summary.get("limit") is not None
    name = f"{summary['stage']} {rc['name']} {summary['split']}" + (" SMOKE" if smoke else "")
    params = {**flat_params(rc, "run."),
              **flat_params({k: summary[k] for k in ("stage", "split", "limit", "git_sha", "manifest_sha256",
                                                    "questions_sha256", "embedder", "table_config",
                                                    "cleaning_version")}),
              **flat_params(summary.get("parser", {}), "parser."),
              **flat_params(summary.get("retriever_params", {}), "retriever.")}
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"stage": summary["stage"], "split": summary["split"], "smoke": str(smoke),
                         "chunker": rc["chunker"], "retriever": rc["retriever"],
                         "rerank": str(rc.get("rerank", False))})
        mlflow.log_params(params)
        mlflow.log_metrics(retrieval_metrics(summary))
        for f in ("summary.json", "questions.jsonl"):
            mlflow.log_artifact(str(run_dir / f))
        return run.info.run_id
