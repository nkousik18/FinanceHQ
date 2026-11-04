"""
Regression test: adding a document to a session must evict the retriever's
cached index for that session.

Bug (fixed in R0): run_pipeline merged new chunks into the S3 index but never
called evict_session(), so queries kept searching the stale in-memory index
until the process restarted.

All external stages (S3, Textract, cleaner, validator, chunker, indexer) are
mocked; only the orchestration in run_pipeline is exercised.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.pipeline import pipeline as pipeline_mod
from app.pipeline.pipeline import DocStatus, run_pipeline
from app.retrieval import retriever as retriever_mod


SESSION = "sess-evict"
OTHER_SESSION = "sess-untouched"


@pytest.fixture
def mocked_stages():
    """Patch every external stage so run_pipeline runs fully offline."""
    extraction = MagicMock(total_pages=2)
    report = MagicMock(passed=True, issues=[])
    chunk = MagicMock()
    chunk.to_dict.return_value = {"chunk_id": "x", "text": "t"}

    with patch.object(pipeline_mod, "get_s3_client", return_value=MagicMock()), \
         patch.object(pipeline_mod, "_read_status", side_effect=pipeline_mod.S3Error("missing")), \
         patch.object(pipeline_mod, "extract_document", return_value=extraction), \
         patch.object(pipeline_mod, "clean_extraction", return_value=MagicMock()), \
         patch.object(pipeline_mod, "validate_extraction", return_value=report), \
         patch.object(pipeline_mod, "chunk_document", return_value=[chunk, chunk]), \
         patch.object(pipeline_mod, "merge_into_session_index", return_value=7) as merge:
        yield merge


@pytest.fixture
def warm_cache():
    """Pre-populate the retriever cache as if both sessions had been queried."""
    retriever_mod._SESSION_CACHE[SESSION] = ("stale-index", ["stale-chunk"])
    retriever_mod._SESSION_CACHE[OTHER_SESSION] = ("other-index", ["other-chunk"])
    yield
    retriever_mod._SESSION_CACHE.pop(SESSION, None)
    retriever_mod._SESSION_CACHE.pop(OTHER_SESSION, None)


def _doc(record):
    return record.documents[-1]


def test_successful_upload_evicts_session_cache(mocked_stages, warm_cache):
    record = run_pipeline(b"%PDF-fake", session_id=SESSION, filename="a.pdf")

    assert _doc(record).status == DocStatus.READY
    assert SESSION not in retriever_mod._SESSION_CACHE


def test_eviction_only_touches_the_uploaded_session(mocked_stages, warm_cache):
    run_pipeline(b"%PDF-fake", session_id=SESSION, filename="a.pdf")

    assert retriever_mod._SESSION_CACHE[OTHER_SESSION] == ("other-index", ["other-chunk"])


def test_eviction_happens_after_merge(mocked_stages, warm_cache):
    """The cache must be dropped after the merged index is written, not before."""
    seen_at_merge = {}

    def _merge(session_id, chunks):
        seen_at_merge["cached"] = session_id in retriever_mod._SESSION_CACHE
        return 7

    mocked_stages.side_effect = _merge
    run_pipeline(b"%PDF-fake", session_id=SESSION, filename="a.pdf")

    assert seen_at_merge["cached"] is True
    assert SESSION not in retriever_mod._SESSION_CACHE


def test_failed_merge_leaves_cache_intact(mocked_stages, warm_cache):
    """If indexing fails, the old index is still the latest, so keep it cached."""
    mocked_stages.side_effect = pipeline_mod.IndexingError("boom")

    record = run_pipeline(b"%PDF-fake", session_id=SESSION, filename="a.pdf")

    assert _doc(record).status == DocStatus.FAILED
    assert retriever_mod._SESSION_CACHE[SESSION] == ("stale-index", ["stale-chunk"])
