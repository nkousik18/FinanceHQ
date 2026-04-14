"""
Tests for the multi-doc session endpoints.

    POST /sessions
    POST /sessions/{session_id}/documents
    GET  /sessions/{session_id}/status
"""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch, call

import pytest
from fastapi.testclient import TestClient

from main import app
from app.pipeline.pipeline import SessionStatus, DocStatus

client = TestClient(app)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pdf(size: int = 1024) -> bytes:
    return b"%PDF-1.4 " + b"x" * size


def _session_record(
    session_id: str = "sess-1",
    status: str = "READY",
    docs: list[dict] | None = None,
) -> dict:
    return {
        "session_id": session_id,
        "name": "Test Session",
        "status": status,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:05:00+00:00",
        "total_pages": 4,
        "total_chunks": 20,
        "documents": docs or [
            {
                "doc_id": "doc-1",
                "filename": "loan.pdf",
                "status": "READY",
                "pages": 4,
                "chunks": 20,
                "uploaded_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:05:00+00:00",
                "error": None,
            }
        ],
        "error": None,
    }


# ------------------------------------------------------------------
# POST /sessions
# ------------------------------------------------------------------

class TestCreateSession:
    @patch("app.api.sessions.create_session")
    def test_creates_session_returns_201(self, mock_create):
        from app.pipeline.pipeline import StatusRecord
        mock_create.return_value = StatusRecord(
            session_id="new-sess-id",
            status=SessionStatus.EMPTY,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            name="My Loans",
        )

        resp = client.post("/sessions", json={"name": "My Loans"})

        assert resp.status_code == 201
        body = resp.json()
        assert body["session_id"] == "new-sess-id"
        assert body["status"] == "EMPTY"
        assert body["name"] == "My Loans"

    @patch("app.api.sessions.create_session")
    def test_name_is_optional(self, mock_create):
        from app.pipeline.pipeline import StatusRecord
        mock_create.return_value = StatusRecord(
            session_id="no-name-sess",
            status=SessionStatus.EMPTY,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            name="",
        )

        resp = client.post("/sessions", json={})
        assert resp.status_code == 201
        mock_create.assert_called_once_with(name="")

    @patch("app.api.sessions.create_session")
    def test_name_forwarded_to_create_session(self, mock_create):
        from app.pipeline.pipeline import StatusRecord
        mock_create.return_value = StatusRecord(
            session_id="s",
            status=SessionStatus.EMPTY,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            name="Home Loan Comparison",
        )

        client.post("/sessions", json={"name": "Home Loan Comparison"})
        mock_create.assert_called_once_with(name="Home Loan Comparison")


# ------------------------------------------------------------------
# POST /sessions/{session_id}/documents
# ------------------------------------------------------------------

class TestUploadDocument:
    @patch("app.api.sessions._run_pipeline_bg")
    @patch("app.api.sessions._write_status")
    @patch("app.api.sessions._read_status")
    def test_returns_202_with_doc_id(self, mock_read, mock_write, mock_bg):
        from app.pipeline.pipeline import StatusRecord
        mock_read.return_value = StatusRecord(
            session_id="sess-1",
            status=SessionStatus.EMPTY,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )

        resp = client.post(
            "/sessions/sess-1/documents",
            files={"file": ("loan.pdf", BytesIO(_pdf()), "application/pdf")},
        )

        assert resp.status_code == 202
        body = resp.json()
        assert body["session_id"] == "sess-1"
        assert "doc_id" in body
        assert len(body["doc_id"]) == 36   # UUID
        assert body["status"] == "PENDING"

    @patch("app.api.sessions._run_pipeline_bg")
    @patch("app.api.sessions._write_status")
    @patch("app.api.sessions._read_status")
    def test_pipeline_scheduled_in_background(self, mock_read, mock_write, mock_bg):
        from app.pipeline.pipeline import StatusRecord
        mock_read.return_value = StatusRecord(
            session_id="sess-1",
            status=SessionStatus.EMPTY,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )

        resp = client.post(
            "/sessions/sess-1/documents",
            files={"file": ("loan.pdf", BytesIO(_pdf()), "application/pdf")},
        )

        assert resp.status_code == 202
        mock_bg.assert_called_once()
        kw = mock_bg.call_args.kwargs
        assert kw["session_id"] == "sess-1"
        assert kw["filename"] == "loan.pdf"
        assert isinstance(kw["pdf_bytes"], bytes)

    @patch("app.api.sessions._read_status")
    def test_unknown_session_returns_404(self, mock_read):
        mock_read.side_effect = S3Error("not found")

        resp = client.post(
            "/sessions/nonexistent/documents",
            files={"file": ("loan.pdf", BytesIO(_pdf()), "application/pdf")},
        )
        assert resp.status_code == 404

    def test_non_pdf_returns_415(self):
        with patch("app.api.sessions._read_status"):
            resp = client.post(
                "/sessions/sess-1/documents",
                files={"file": ("doc.txt", BytesIO(b"hello"), "text/plain")},
            )
        assert resp.status_code == 415

    def test_empty_file_returns_400(self):
        with patch("app.api.sessions._read_status") as mock_read:
            from app.pipeline.pipeline import StatusRecord
            mock_read.return_value = StatusRecord(
                session_id="sess-1",
                status=SessionStatus.EMPTY,
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
            )
            resp = client.post(
                "/sessions/sess-1/documents",
                files={"file": ("empty.pdf", BytesIO(b""), "application/pdf")},
            )
        assert resp.status_code == 400

    def test_oversized_file_returns_413(self):
        with patch("app.api.sessions._read_status") as mock_read:
            from app.pipeline.pipeline import StatusRecord
            mock_read.return_value = StatusRecord(
                session_id="sess-1",
                status=SessionStatus.EMPTY,
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
            )
            big = b"x" * (21 * 1024 * 1024)
            resp = client.post(
                "/sessions/sess-1/documents",
                files={"file": ("big.pdf", BytesIO(big), "application/pdf")},
            )
        assert resp.status_code == 413


# ------------------------------------------------------------------
# GET /sessions/{session_id}/status
# ------------------------------------------------------------------

class TestGetSessionStatus:
    @patch("app.api.sessions._read_status")
    def test_ready_session_with_one_doc(self, mock_read):
        from app.pipeline.pipeline import StatusRecord, DocRecord
        record = StatusRecord(
            session_id="sess-1",
            status=SessionStatus.READY,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:05:00+00:00",
            name="My Loans",
            total_pages=4,
            total_chunks=20,
            documents=[
                DocRecord(
                    doc_id="doc-1",
                    filename="loan.pdf",
                    status=DocStatus.READY,
                    uploaded_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:05:00+00:00",
                    pages=4,
                    chunks=20,
                )
            ],
        )
        mock_read.return_value = record

        resp = client.get("/sessions/sess-1/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "READY"
        assert body["total_chunks"] == 20
        assert len(body["documents"]) == 1
        assert body["documents"][0]["status"] == "READY"
        assert body["documents"][0]["filename"] == "loan.pdf"

    @patch("app.api.sessions._read_status")
    def test_processing_session_with_two_docs(self, mock_read):
        from app.pipeline.pipeline import StatusRecord, DocRecord
        record = StatusRecord(
            session_id="sess-2",
            status=SessionStatus.PROCESSING,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:03:00+00:00",
            documents=[
                DocRecord(
                    doc_id="doc-1",
                    filename="loan1.pdf",
                    status=DocStatus.READY,
                    uploaded_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:02:00+00:00",
                    pages=2, chunks=10,
                ),
                DocRecord(
                    doc_id="doc-2",
                    filename="loan2.pdf",
                    status=DocStatus.EXTRACTING,
                    uploaded_at="2026-01-01T00:02:00+00:00",
                    updated_at="2026-01-01T00:03:00+00:00",
                ),
            ],
        )
        mock_read.return_value = record

        resp = client.get("/sessions/sess-2/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "PROCESSING"
        assert len(body["documents"]) == 2

    @patch("app.api.sessions._read_status")
    def test_unknown_session_returns_404(self, mock_read):
        mock_read.side_effect = S3Error("not found")
        resp = client.get("/sessions/nonexistent/status")
        assert resp.status_code == 404

    @patch("app.api.sessions._read_status")
    def test_empty_session_has_no_docs(self, mock_read):
        from app.pipeline.pipeline import StatusRecord
        mock_read.return_value = StatusRecord(
            session_id="empty",
            status=SessionStatus.EMPTY,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        resp = client.get("/sessions/empty/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "EMPTY"
        assert resp.json()["documents"] == []


# ------------------------------------------------------------------
# Pipeline orchestrator unit tests
# ------------------------------------------------------------------

def _empty_session(session_id: str = "sess-1") -> "StatusRecord":
    from app.pipeline.pipeline import StatusRecord
    return StatusRecord(
        session_id=session_id,
        status=SessionStatus.EMPTY,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


class TestPipelineMultiDoc:
    @patch("app.pipeline.pipeline._write_status")
    @patch("app.pipeline.pipeline._read_status")
    @patch("app.pipeline.pipeline.merge_into_session_index")
    @patch("app.pipeline.pipeline.chunk_document")
    @patch("app.pipeline.pipeline.validate_extraction")
    @patch("app.pipeline.pipeline.clean_extraction")
    @patch("app.pipeline.pipeline.extract_document")
    @patch("app.pipeline.pipeline.get_s3_client")
    def test_doc_reaches_ready(
        self, mock_s3, mock_extract, mock_clean, mock_validate,
        mock_chunk, mock_merge, mock_read, mock_write
    ):
        from app.pipeline.pipeline import run_pipeline

        mock_s3.return_value = MagicMock()
        mock_read.return_value = _empty_session("s1")
        mock_extract.return_value = MagicMock(total_pages=3)
        mock_clean.return_value = MagicMock()
        mock_validate.return_value = MagicMock(passed=True, issues=[])
        chunk = MagicMock()
        chunk.chunk_id = "old-id"
        chunk.to_dict.return_value = {"chunk_id": "old-id", "text": "t"}
        mock_chunk.return_value = [chunk] * 5
        mock_merge.return_value = 5

        record = run_pipeline(b"pdf bytes", session_id="s1", filename="doc.pdf")

        assert len(record.documents) == 1
        assert record.documents[0].status == DocStatus.READY
        assert record.status == SessionStatus.READY

    @patch("app.pipeline.pipeline._write_status")
    @patch("app.pipeline.pipeline._read_status")
    @patch("app.pipeline.pipeline.extract_document")
    @patch("app.pipeline.pipeline.get_s3_client")
    def test_extraction_failure_sets_doc_failed(
        self, mock_s3, mock_extract, mock_read, mock_write
    ):
        from app.pipeline.pipeline import run_pipeline
        from app.pipeline.extractor import ExtractionError

        mock_s3.return_value = MagicMock()
        mock_read.return_value = _empty_session("s1")
        mock_extract.side_effect = ExtractionError("Textract timeout")

        record = run_pipeline(b"pdf bytes", session_id="s1")

        assert record.documents[0].status == DocStatus.FAILED
        assert "Textract timeout" in record.documents[0].error
        assert record.status == SessionStatus.FAILED

    @patch("app.pipeline.pipeline._write_status")
    @patch("app.pipeline.pipeline._read_status")
    @patch("app.pipeline.pipeline.merge_into_session_index")
    @patch("app.pipeline.pipeline.chunk_document")
    @patch("app.pipeline.pipeline.validate_extraction")
    @patch("app.pipeline.pipeline.clean_extraction")
    @patch("app.pipeline.pipeline.extract_document")
    @patch("app.pipeline.pipeline.get_s3_client")
    def test_two_docs_merged_into_one_session(
        self, mock_s3, mock_extract, mock_clean, mock_validate,
        mock_chunk, mock_merge, mock_read, mock_write
    ):
        from app.pipeline.pipeline import run_pipeline, StatusRecord, DocRecord

        mock_s3.return_value = MagicMock()
        mock_extract.return_value = MagicMock(total_pages=2)
        mock_clean.return_value = MagicMock()
        mock_validate.return_value = MagicMock(passed=True, issues=[])
        chunk = MagicMock()
        chunk.chunk_id = "x"
        chunk.to_dict.return_value = {"chunk_id": "x", "text": "t"}
        mock_chunk.return_value = [chunk] * 8
        mock_merge.side_effect = [8, 16]

        # First call returns empty session; second call returns session with doc1 already READY
        session_with_doc1 = StatusRecord(
            session_id="s1",
            status=SessionStatus.READY,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            documents=[
                DocRecord(
                    doc_id="doc-1", filename="doc1.pdf",
                    status=DocStatus.READY,
                    uploaded_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:02:00+00:00",
                    pages=2, chunks=8,
                )
            ],
        )
        mock_read.side_effect = [_empty_session("s1"), session_with_doc1]

        run_pipeline(b"pdf1", session_id="s1", filename="doc1.pdf")
        record = run_pipeline(b"pdf2", session_id="s1", filename="doc2.pdf")

        assert len(record.documents) == 2
        assert all(d.status == DocStatus.READY for d in record.documents)
        assert record.status == SessionStatus.READY
        assert mock_merge.call_count == 2

    @patch("app.pipeline.pipeline._write_status")
    @patch("app.pipeline.pipeline._read_status")
    @patch("app.pipeline.pipeline.merge_into_session_index")
    @patch("app.pipeline.pipeline.chunk_document")
    @patch("app.pipeline.pipeline.validate_extraction")
    @patch("app.pipeline.pipeline.clean_extraction")
    @patch("app.pipeline.pipeline.extract_document")
    @patch("app.pipeline.pipeline.get_s3_client")
    def test_one_failed_one_ready_session_is_ready(
        self, mock_s3, mock_extract, mock_clean, mock_validate,
        mock_chunk, mock_merge, mock_read, mock_write
    ):
        """Session stays READY as long as at least one doc succeeded."""
        from app.pipeline.pipeline import run_pipeline, StatusRecord, DocRecord
        from app.pipeline.extractor import ExtractionError

        mock_s3.return_value = MagicMock()
        chunk = MagicMock()
        chunk.chunk_id = "x"

        session_with_ready_doc = StatusRecord(
            session_id="s1",
            status=SessionStatus.READY,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            documents=[
                DocRecord(
                    doc_id="d1", filename="doc1.pdf",
                    status=DocStatus.READY,
                    uploaded_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:02:00+00:00",
                    pages=2, chunks=5,
                )
            ],
        )
        mock_read.side_effect = [_empty_session("s1"), session_with_ready_doc]

        mock_extract.return_value = MagicMock(total_pages=2)
        mock_clean.return_value = MagicMock()
        mock_validate.return_value = MagicMock(passed=True, issues=[])
        mock_chunk.return_value = [chunk] * 5
        mock_merge.return_value = 5

        run_pipeline(b"pdf1", session_id="s1", filename="doc1.pdf")

        mock_extract.side_effect = ExtractionError("bad PDF")
        record = run_pipeline(b"pdf2", session_id="s1", filename="doc2.pdf")

        doc_statuses = {d.filename: d.status for d in record.documents}
        assert doc_statuses["doc1.pdf"] == DocStatus.READY
        assert doc_statuses["doc2.pdf"] == DocStatus.FAILED
        assert record.status == SessionStatus.READY

    @patch("app.pipeline.pipeline._write_status")
    @patch("app.pipeline.pipeline._read_status")
    @patch("app.pipeline.pipeline.merge_into_session_index")
    @patch("app.pipeline.pipeline.chunk_document")
    @patch("app.pipeline.pipeline.validate_extraction")
    @patch("app.pipeline.pipeline.clean_extraction")
    @patch("app.pipeline.pipeline.extract_document")
    @patch("app.pipeline.pipeline.get_s3_client")
    def test_chunk_ids_stamped_with_doc_id(
        self, mock_s3, mock_extract, mock_clean, mock_validate,
        mock_chunk, mock_merge, mock_read, mock_write
    ):
        """Each chunk's ID must include the doc_id to avoid collisions across docs."""
        from app.pipeline.pipeline import run_pipeline

        mock_s3.return_value = MagicMock()
        mock_read.return_value = _empty_session("s1")
        mock_extract.return_value = MagicMock(total_pages=1)
        mock_clean.return_value = MagicMock()
        mock_validate.return_value = MagicMock(passed=True, issues=[])

        chunks = [MagicMock(chunk_id="original") for _ in range(3)]
        mock_chunk.return_value = chunks
        mock_merge.return_value = 3

        doc_id = "my-doc-id"
        run_pipeline(b"pdf", session_id="s1", filename="f.pdf", doc_id=doc_id)

        for i, chunk in enumerate(chunks):
            assert doc_id in chunk.chunk_id
            assert str(i) in chunk.chunk_id


# ------------------------------------------------------------------
# Merge index unit tests
# ------------------------------------------------------------------

class TestMergeIntoSessionIndex:
    @patch("app.pipeline.indexer.get_s3_client")
    @patch("app.pipeline.indexer.get_embedder")
    def test_first_doc_creates_new_index(self, mock_embedder, mock_s3):
        import numpy as np
        from app.pipeline.indexer import merge_into_session_index
        from app.pipeline.chunker import Chunk

        mock_s3_inst = MagicMock()
        mock_s3_inst.exists.return_value = False   # no existing index
        mock_s3.return_value = mock_s3_inst

        embedder = MagicMock()
        embedder.embed.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_embedder.return_value = embedder

        chunks = [
            Chunk(chunk_id=f"s-d-{i}", session_id="s", index=i, text=f"text {i}",
                  word_count=2, page=1, section="A", token_estimate=3)
            for i in range(3)
        ]

        total = merge_into_session_index("session-1", chunks)
        assert total == 3

    @patch("app.pipeline.indexer.get_s3_client")
    @patch("app.pipeline.indexer.get_embedder")
    def test_second_doc_appends_to_existing(self, mock_embedder, mock_s3):
        import numpy as np
        import io as _io
        import json as _json
        from app.pipeline.indexer import merge_into_session_index
        from app.pipeline.chunker import Chunk

        existing_chunks = [
            Chunk(chunk_id=f"s-d1-{i}", session_id="s", index=i, text=f"old {i}",
                  word_count=2, page=1, section="A", token_estimate=3)
            for i in range(5)
        ]
        existing_emb = np.random.rand(5, 384).astype(np.float32)

        npy_buf = _io.BytesIO()
        np.save(npy_buf, existing_emb)
        npy_bytes = npy_buf.getvalue()

        mock_s3_inst = MagicMock()
        mock_s3_inst.exists.return_value = True
        mock_s3_inst.download_text.return_value = _json.dumps([c.to_dict() for c in existing_chunks])
        mock_s3_inst.download_bytes.return_value = npy_bytes
        mock_s3.return_value = mock_s3_inst

        embedder = MagicMock()
        embedder.embed.return_value = np.random.rand(3, 384).astype(np.float32)
        mock_embedder.return_value = embedder

        new_chunks = [
            Chunk(chunk_id=f"s-d2-{i}", session_id="s", index=i, text=f"new {i}",
                  word_count=2, page=1, section="B", token_estimate=3)
            for i in range(3)
        ]

        total = merge_into_session_index("s", new_chunks)
        assert total == 8   # 5 existing + 3 new


# Fix import
from app.storage.s3_client import S3Error
