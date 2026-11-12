"""
F9 (scope v2): git must never track raw corpus PDFs, caches, the local MLflow
store, the spend ledger, or .env files. Fails the build if any slip in.

Allowed exceptions: PDFs under evals/fixtures/ci/ (small CI fixtures, §9.7).
"""
import fnmatch
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

FORBIDDEN = [
    "data/*",
    "mlruns/*",
    "mlartifacts/*",
    "*.env",
    ".env",
    ".env.*",
    "*/spend_ledger*",
    "evals/*.pdf",
    "evals/**/*.pdf",
    "*.index",
    "*.npy",
]
ALLOWED = [
    ".env.example",
    "*/.env.example",
    "evals/fixtures/ci/*",
]


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [line for line in out.splitlines() if line]


def violations(paths: list[str]) -> list[str]:
    bad = []
    for p in paths:
        if any(fnmatch.fnmatch(p, a) for a in ALLOWED):
            continue
        if any(fnmatch.fnmatch(p, f) for f in FORBIDDEN):
            bad.append(p)
    return bad


def test_no_private_or_bulky_data_tracked():
    bad = violations(_tracked_files())
    assert not bad, f"These files must not be tracked (scope v2 D11/F9): {bad}"


@pytest.mark.parametrize("path", [
    "data/corpus/deal_01_ca.pdf",
    "data/mlflow/mlflow.db",
    "evals/corpus/raw/x.pdf",
    ".env",
    "fastapi_service/.env",
    "data/spend_ledger.json",
    "chunks/faiss.index",
])
def test_rules_catch_forbidden_paths(path):
    assert violations([path]) == [path]


@pytest.mark.parametrize("path", [
    "evals/fixtures/ci/cfpb_le_01.pdf",
    "deploy/env.example",
    ".env.example",
    "evals/golden/ingest/images/p01.png",
    "evals/corpus/manifest.csv",
])
def test_rules_allow_committed_paths(path):
    assert violations([path]) == []
