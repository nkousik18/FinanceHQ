"""Fixes for the empty scans (S-2026-11-12-13 image filings, S-2026-11-12-14 Tesseract orientation)."""
import os
import subprocess

from evals.corpus.render import IMAGE_FILING_MIN_IMAGES, image_sources, is_image_filing, sanitize_html
from evals.parsers.config import CONFIGS, TESSERACT_NO_OSD, cache_key

PAGE = '<html><head></head><body><IMG src="p001.jpg" title="slide1"><font style="color:white">text</font>' \
       '<img src=\'logo.png\'></body></html>'


def test_default_still_strips_images():
    html, stats = sanitize_html(PAGE.encode())
    assert "<img" not in html.lower() and stats["images_removed"] == 2
    assert "img-src" not in html


def test_embeds_given_images_and_strips_the_rest():
    html, stats = sanitize_html(PAGE.encode(), images={"p001.jpg": b"\xff\xd8jpeg"})
    assert 'src="data:image/jpeg;base64,' in html
    assert "logo.png" not in html
    assert stats == {"scripts_removed": 0, "images_embedded": 1, "images_removed": 1}
    assert "img-src data:" in html and "max-width: 100%" in html


def test_image_filing_threshold():
    one = PAGE.encode()
    many = ("".join(f'<img src="p{i:03d}.jpg">' for i in range(IMAGE_FILING_MIN_IMAGES))).encode()
    assert image_sources(one) == ["p001.jpg", "logo.png"]
    assert not is_image_filing(one) and is_image_filing(many)


def test_orientation_setting_only_changes_tesseract_config_keys():
    assert CONFIGS["P1"].as_dict()["ocr_orientation"] == "fixed_0"
    assert "ocr_orientation" not in CONFIGS["P1V"].as_dict()      # default omitted: Vision cache keys unchanged
    assert cache_key("x", "v", CONFIGS["P1"]) != cache_key("x", "v", CONFIGS["P2"])


def test_shim_answers_osd_with_zero_and_passes_other_calls_through():
    osd = subprocess.run([str(TESSERACT_NO_OSD), "--psm", "0", "-l", "osd", "img.png", "stdout"],
                         capture_output=True, text=True, check=True).stdout
    assert "Orientation in degrees: 0" in osd and "Rotate: 0" in osd
    env = {**os.environ, "TESSERACT_REAL": "echo"}
    out = subprocess.run([str(TESSERACT_NO_OSD), "-l", "eng", "img.png", "stdout", "tsv"],
                         capture_output=True, text=True, check=True, env=env).stdout
    assert out.strip() == "-l eng img.png stdout tsv"
