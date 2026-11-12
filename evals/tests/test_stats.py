"""F8 tests for evals/stats.py."""
import numpy as np
import pytest

from evals.stats import bootstrap_ci, holm, paired_diff_ci, ratio_ci


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    x = [0.1, 0.2, 0.3, 0.4, 0.5] * 10
    a, b = bootstrap_ci(x, seed=1), bootstrap_ci(x, seed=1)
    assert a == b
    assert a["lo"] < 0.3 < a["hi"] and a["point"] == pytest.approx(0.3) and a["n"] == 50


def test_bootstrap_median_and_none_dropped():
    out = bootstrap_ci([1, 2, 3, None, 100], stat=np.median)
    assert out["point"] == 2.5 and out["n"] == 4


def test_ratio_ci_pools_counts():
    out = ratio_ci([3, 1], [4, 4])          # 4/8 pooled, not mean of 0.75 and 0.25 weighted otherwise
    assert out["point"] == 0.5
    assert ratio_ci([0], [0])["point"] is None


def test_paired_diff_detects_consistent_improvement():
    a = [0.5, 0.6, 0.7, 0.55, 0.65] * 6
    b = [x + 0.1 for x in a]
    out = paired_diff_ci(a, b)
    assert out["diff"] == pytest.approx(0.1) and out["better"] == "b" and out["p"] < 0.05


def test_paired_diff_no_difference():
    rng = np.random.default_rng(0)
    a = list(rng.normal(0.5, 0.1, 40))
    b = [x + d for x, d in zip(a, rng.normal(0, 0.1, 40))]
    out = paired_diff_ci(a, b)
    assert out["lo"] < 0 < out["hi"] and out["better"] is None


def test_paired_diff_skips_undefined_units():
    assert paired_diff_ci([0.5, None, 0.7], [0.6, 0.9, None])["n"] == 1


def test_holm_known_values():
    # p = [0.01, 0.04, 0.03] -> sorted 0.01*3=0.03, 0.03*2=0.06, 0.04*1 -> max(0.06, 0.04)=0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_h6_pairs_each_scan_with_its_source_page():
    from evals.ingest_metrics.run_s0 import h6
    recs = [
        {"config": "P1", "part": "edgar", "doc_id": "d", "page": 3, "ref_usable": True, "cer": 0.01},
        {"config": "P1", "part": "consumer", "doc_id": "c", "page": 2, "ref_usable": True, "cer": 0.40},  # not a source
        {"config": "P1", "part": "simulated_scan", "doc_id": "scan_01", "page": 1, "ref_usable": True,
         "cer": 0.05, "ref_source": "d:p3"},
    ]
    out = h6(recs)["P1"]
    assert out["pairs"] == 1 and out["source_cer_median"] == 0.01 and out["supported"] is True
