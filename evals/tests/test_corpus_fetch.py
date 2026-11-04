"""Tests for manifest I/O, checksum handling and HTML sanitizing (no network, no Chrome)."""
import pytest

from evals.corpus.fetch import ChecksumError, sha256, verify_or_freeze
from evals.corpus.manifest import DocRow, edgar_rows, read_manifest, write_manifest
from evals.corpus.render import sanitize_html, source_page_count


class TestSanitize:
    RAW = (b"<html><head><title>x</title></head><body>"
           b"<img src='logo.jpg'><script>alert(1)</script><p>CREDIT AGREEMENT</p></body></html>")

    def test_removes_images_and_scripts(self):
        html, stats = sanitize_html(self.RAW)
        assert "<img" not in html.lower() and "<script" not in html.lower()
        assert stats == {"images_removed": 1, "scripts_removed": 1}
        assert "CREDIT AGREEMENT" in html

    def test_blocks_network_and_sets_page(self):
        html, _ = sanitize_html(self.RAW, margin_in=0.5, zoom=0.9)
        assert "default-src 'none'" in html
        assert "margin: 0.5in" in html and "zoom: 0.9" in html

    def test_works_without_head(self):
        html, _ = sanitize_html(b"<p>no head</p>")
        assert html.startswith('<meta charset="utf-8">')


@pytest.mark.parametrize("raw,expected", [
    (b"<p>one page</p>", 1),
    (b'<p style="page-break-after:always">a</p><p>b</p>', 2),
    (b'<div style="break-before: page">a</div><div style="PAGE-BREAK-BEFORE: always">b</div>', 3),
    (b'<p style="page-break-inside:avoid">not a break</p>', 1),
])
def test_source_page_count(raw, expected):
    assert source_page_count(raw) == expected


class TestChecksum:
    def test_freeze_records_digest(self):
        row = DocRow(doc_id="d", part="edgar")
        verify_or_freeze(row, b"abc", freeze=True)
        assert row.source_sha256 == sha256(b"abc")

    def test_mismatch_raises(self):
        row = DocRow(doc_id="d", part="edgar", source_sha256=sha256(b"abc"))
        with pytest.raises(ChecksumError):
            verify_or_freeze(row, b"changed", freeze=True)

    def test_unfrozen_without_flag_raises(self):
        with pytest.raises(ChecksumError):
            verify_or_freeze(DocRow(doc_id="d", part="edgar"), b"abc", freeze=False)


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "m.csv"
    rows = [DocRow(doc_id="b", part="edgar", company="Acme, Inc."), DocRow(doc_id="a", part="consumer")]
    write_manifest(rows, path)
    back = read_manifest(path)
    assert [r.doc_id for r in back] == ["a", "b"]          # sorted by part, doc_id
    assert back[1].company == "Acme, Inc."                 # comma survives CSV quoting


def test_edgar_rows_ids_from_amendment_titles():
    selection = {"deals": [{"deal_id": "deal_01", "cik": "1", "name": "Acme"}]}
    doc = {"accession": "a", "file_date": "2020-01-01", "form": "8-K", "file_type": "EX-10.1", "url": "u"}
    candidates = {"candidates": [{
        "cik": "1",
        "original": doc,
        "amendments": [{**doc, "opening": "SECOND AMENDMENT TO CREDIT AGREEMENT"},
                       {**doc, "opening": "THIRD AMENDMENT TO CREDIT AGREEMENT"}],
    }]}
    ids = [r.doc_id for r in edgar_rows(selection, candidates)]
    assert ids == ["deal_01_ca", "deal_01_am2", "deal_01_am3"]
