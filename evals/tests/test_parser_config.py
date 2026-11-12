"""Tests for evals/parsers/config.py (cache keys and lookup). Stdlib only."""
import json

import pytest

from evals.parsers.config import P1, P2, P1V, P2V, cache_key, cache_path, downstream_config, is_cached, load_cached


def test_cache_key_stable_and_sensitive():
    k = cache_key("abc", "2.147.0", P1)
    assert k == cache_key("abc", "2.147.0", P1)
    assert k != cache_key("abd", "2.147.0", P1)        # document changed
    assert k != cache_key("abc", "2.148.0", P1)        # parser upgraded
    assert k != cache_key("abc", "2.147.0", P2)        # config changed


def test_is_cached_requires_matching_key(tmp_path):
    key = cache_key("abc", "2.147.0", P1)
    path = cache_path(P1, "doc1", root=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"meta": {"cache_key": key, "doc_id": "doc1"}, "document": {}}))

    assert is_cached(P1, "doc1", key, root=tmp_path)
    assert not is_cached(P1, "doc1", cache_key("other", "2.147.0", P1), root=tmp_path)
    assert not is_cached(P1, "missing", key, root=tmp_path)
    assert load_cached(P1, "doc1", root=tmp_path)["meta"]["doc_id"] == "doc1"


def test_configs_differ_only_in_table_mode():
    a, b = P1.as_dict(), P2.as_dict()   # P1V is a scans-only OCR diagnostic, compared separately
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"name", "table_mode"}


def test_downstream_uses_vision_text_for_scans_only():
    assert downstream_config("P1", "simulated_scan") == "P1V"
    assert downstream_config("P2", "simulated_scan") == "P2V"
    assert downstream_config("P2", "edgar") == "P2"
    assert downstream_config("P1", "consumer") == "P1"
    with pytest.raises(ValueError):
        downstream_config("P1V", "edgar")


def test_vision_variants_match_their_table_mode():
    assert (P1V.table_mode, P2V.table_mode) == (P1.table_mode, P2.table_mode)
    assert P1V.ocr_engine == P2V.ocr_engine == "ocrmac"
