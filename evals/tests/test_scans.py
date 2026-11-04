"""Tests for evals/corpus/scans.py (simulated scans). Uses small generated PDFs only."""
import random

import fitz
import pytest

from evals.corpus import scans
from evals.corpus.manifest import DocRow


def _page_with_text(words: int) -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page()
    text = " ".join(f"word{i}" for i in range(words))
    page.insert_textbox(fitz.Rect(36, 36, 576, 756), text, fontsize=9)
    return page


class TestDegrade:
    def test_deterministic_for_same_seed(self):
        img = scans.rasterize(_page_with_text(50), dpi=100)
        d = scans.LEVELS["medium"]
        a, pa = scans.degrade(img, d, random.Random(7))
        b, pb = scans.degrade(img, d, random.Random(7))
        assert a == b and pa == pb

    def test_skew_within_level_bounds(self):
        img = scans.rasterize(_page_with_text(10), dpi=72)
        for level, d in scans.LEVELS.items():
            _, applied = scans.degrade(img, d, random.Random(1))
            assert abs(applied["skew_deg"]) <= d.max_skew_deg

    def test_levels_get_progressively_worse(self):
        light, medium, heavy = (scans.LEVELS[k] for k in scans.LEVEL_ORDER)
        assert light.dpi > medium.dpi > heavy.dpi
        assert light.jpeg_quality > medium.jpeg_quality > heavy.jpeg_quality
        assert light.noise_sigma < medium.noise_sigma < heavy.noise_sigma


def test_image_pdf_has_no_text_layer(tmp_path):
    page = _page_with_text(100)
    jpeg, _ = scans.degrade(scans.rasterize(page, 100), scans.LEVELS["light"], random.Random(0))
    out = tmp_path / "scan.pdf"
    scans.image_pdf(jpeg, page.rect, out)
    with fitz.open(out) as d:
        assert d.page_count == 1
        assert d[0].get_text().strip() == ""
        assert len(d[0].get_images()) == 1


@pytest.mark.parametrize("text,expected", [
    ("By: /s/ A\nName: A\nTitle: CFO\nBy: /s/ B", True),
    ("The Borrower shall repay the Loans. Title: none", False),
])
def test_signature_page_detection(text, expected):
    assert scans.is_signature_page(text) is expected


def test_select_pages_respects_quotas_caps_and_min_words(tmp_path):
    rows = []
    for i in range(4):
        doc = fitz.open()
        for words in (200, 200, 200, 10):          # last page too short
            p = doc.new_page()
            p.insert_textbox(fitz.Rect(36, 36, 576, 756), " ".join(["text"] * words), fontsize=8)
        doc.save(tmp_path / f"d{i}.pdf")
        rows.append(DocRow(doc_id=f"d{i}", part="edgar", role="original", pdf_pages="4"))

    picked = scans.select_pages(rows, seed=1, quotas={"edgar_prose": 5}, pdf_dir=tmp_path)

    assert len(picked) == 5
    assert all(p["page"] != 4 for p in picked)                 # short page excluded
    per_doc = {}
    for p in picked:
        per_doc[p["source_doc"]] = per_doc.get(p["source_doc"], 0) + 1
    assert max(per_doc.values()) <= scans.MAX_PER_DOC_STRATUM
    assert {p["level"] for p in picked} == {"light", "medium", "heavy"}
