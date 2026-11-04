"""
Task callables for the placeholder LoanDoc ingestion DAG.

STANDALONE PLACEHOLDER (R0). These functions log what each stage would do and
pass a small metadata dict to the next stage. They do not call Docling, the
FastAPI pipeline, the PII module, or any storage. Nothing in the app imports
them.

They are plain Python (no Airflow import), so they can be unit-tested without
installing Airflow. The DAG in ../dags wraps each one in a PythonOperator and
moves the dict between tasks through XCom.

Stage order follows the evals scope v2 corpus flow:
    fetch_corpus -> parse_documents -> clean_documents -> validate_documents
    -> pii_scan -> chunk_documents -> build_index -> write_run_report
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("loandoc.orchestration")

STAGES = [
    "fetch_corpus",
    "parse_documents",
    "clean_documents",
    "validate_documents",
    "pii_scan",
    "chunk_documents",
    "build_index",
    "write_run_report",
]

# Stand-in corpus: what evals/corpus/manifest.csv will list once R1 builds it.
_PLACEHOLDER_MANIFEST = [
    {"doc_id": "deal_01_ca", "part": "edgar", "pages": 180},
    {"doc_id": "deal_01_am1", "part": "edgar", "pages": 14},
    {"doc_id": "cfpb_le_01", "part": "consumer", "pages": 3},
]


def _advance(state: dict[str, Any], stage: str, **updates: Any) -> dict[str, Any]:
    """Return a new state dict with this stage recorded. Never mutates input."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    new = {**state, **updates}
    new["completed_stages"] = [*state.get("completed_stages", []), stage]
    log.info("stage=%s placeholder=true docs=%d", stage, len(new.get("documents", [])))
    return new


def fetch_corpus(run_id: str = "manual") -> dict[str, Any]:
    """Would run evals/corpus/fetch (download + render + checksum). Placeholder: static manifest."""
    return _advance(
        {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat()},
        "fetch_corpus",
        documents=[dict(d) for d in _PLACEHOLDER_MANIFEST],
    )


def parse_documents(state: dict[str, Any]) -> dict[str, Any]:
    """Would parse each PDF with Docling via the parser interface and cache it."""
    docs = [{**d, "parsed": True, "parser": "docling (placeholder)"} for d in state["documents"]]
    return _advance(state, "parse_documents", documents=docs)


def clean_documents(state: dict[str, Any]) -> dict[str, Any]:
    """Would run the 7 cleaning fixes from app/pipeline/cleaner.py."""
    docs = [{**d, "cleaned": True} for d in state["documents"]]
    return _advance(state, "clean_documents", documents=docs)


def validate_documents(state: dict[str, Any]) -> dict[str, Any]:
    """Would run app/pipeline/validator.py; ERROR-level docs are dropped from later stages."""
    docs = [{**d, "validation": "pass"} for d in state["documents"]]
    return _advance(state, "validate_documents", documents=docs)


def pii_scan(state: dict[str, Any]) -> dict[str, Any]:
    """
    Would call app.privacy.pii.redact on each document's cleaned text and
    record counts per PII type. The public evals corpus is expected to have
    none; this stage matters for user uploads.
    """
    docs = [{**d, "pii_counts": {}} for d in state["documents"]]
    return _advance(state, "pii_scan", documents=docs)


def chunk_documents(state: dict[str, Any]) -> dict[str, Any]:
    """Would chunk with the configured chunker (C1 by default)."""
    docs = [{**d, "chunks": 0} for d in state["documents"]]
    return _advance(state, "chunk_documents", documents=docs)


def build_index(state: dict[str, Any]) -> dict[str, Any]:
    """Would embed chunks with MiniLM and build one FAISS index per session."""
    return _advance(state, "build_index", index={"type": "IndexFlatIP", "vectors": 0})


def write_run_report(state: dict[str, Any]) -> dict[str, Any]:
    """Would log the run to local MLflow. Placeholder: summary dict only."""
    summary = {
        "documents": len(state.get("documents", [])),
        "pages": sum(d.get("pages", 0) for d in state.get("documents", [])),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    return _advance(state, "write_run_report", summary=summary)


def run_all(run_id: str = "manual") -> dict[str, Any]:
    """Run every stage in order without Airflow (local smoke check)."""
    state = fetch_corpus(run_id)
    for fn in (parse_documents, clean_documents, validate_documents, pii_scan,
               chunk_documents, build_index, write_run_report):
        state = fn(state)
    return state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    final = run_all()
    print(final["summary"])
