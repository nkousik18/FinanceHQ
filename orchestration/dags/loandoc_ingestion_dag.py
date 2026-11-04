"""
loandoc_ingestion_dag — PLACEHOLDER Airflow DAG (R0).

Not scheduled, not connected to the FastAPI pipeline or the evals. Each task
wraps a stub callable from orchestration/tasks/ingestion_tasks.py and passes
a metadata dict to the next task through XCom.

Layout follows the OCR DAG in the sibling project LoanQA-MLOps
(dags/ocr_pipeline_dag.py): resolve the project root for both local and
Docker runs, import stage functions, one PythonOperator per stage, linear
dependencies. Textract/S3 stages are replaced by the local Docling flow from
the evals scope v2.

Run locally (after `pip install apache-airflow`):
    export AIRFLOW_HOME=$(pwd)/orchestration/.airflow
    export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/orchestration/dags
    airflow dags test loandoc_ingestion_placeholder
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Project root: /opt/project inside an Airflow container, else this repo.
if os.path.exists("/opt/project"):
    PROJECT_ROOT = "/opt/project"
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from orchestration.tasks import ingestion_tasks as t  # noqa: E402

default_args = {
    "owner": "loandoc",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _first(**context):
    return t.fetch_corpus(run_id=context["run_id"])


def _wrap(fn, upstream_task_id):
    """Pull the upstream state from XCom, run the stage, push the new state."""
    def _call(**context):
        state = context["ti"].xcom_pull(task_ids=upstream_task_id)
        return fn(state)
    return _call


with DAG(
    dag_id="loandoc_ingestion_placeholder",
    description="Placeholder: fetch -> parse (Docling) -> clean -> validate -> PII -> chunk -> index -> report",
    default_args=default_args,
    start_date=datetime(2026, 11, 1),
    schedule=None,          # manual trigger only
    catchup=False,
    tags=["loandoc", "placeholder"],
) as dag:

    stages = [("fetch_corpus", _first)]
    for name in t.STAGES[1:]:
        stages.append((name, _wrap(getattr(t, name), stages[-1][0])))

    tasks = [PythonOperator(task_id=name, python_callable=fn) for name, fn in stages]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
