"""
Tests for the placeholder orchestration task callables. No Airflow needed.
Run from repo root:  venv/bin/python -m pytest orchestration/tests -q
"""
import ast
from pathlib import Path

import pytest

from orchestration.tasks import ingestion_tasks as t

DAG_FILE = Path(__file__).resolve().parents[1] / "dags" / "loandoc_ingestion_dag.py"


def test_run_all_completes_every_stage_in_order():
    final = t.run_all(run_id="test")
    assert final["completed_stages"] == t.STAGES
    assert final["run_id"] == "test"


def test_summary_counts_documents_and_pages():
    final = t.run_all()
    assert final["summary"]["documents"] == len(final["documents"])
    assert final["summary"]["pages"] == sum(d["pages"] for d in final["documents"])


def test_stages_do_not_mutate_input_state():
    state = t.fetch_corpus()
    before = {**state, "documents": [dict(d) for d in state["documents"]]}
    t.parse_documents(state)
    assert state == before


def test_each_stage_tags_documents():
    final = t.run_all()
    for d in final["documents"]:
        assert d["parsed"] and d["cleaned"]
        assert d["validation"] == "pass"
        assert d["pii_counts"] == {}


def test_unknown_stage_rejected():
    with pytest.raises(ValueError):
        t._advance({}, "not_a_stage")


def test_dag_file_is_valid_python_and_uses_all_stages():
    """Parse the DAG without importing Airflow."""
    src = DAG_FILE.read_text()
    ast.parse(src)
    assert 'dag_id="loandoc_ingestion_placeholder"' in src
    assert "schedule=None" in src
    assert "t.STAGES" in src
