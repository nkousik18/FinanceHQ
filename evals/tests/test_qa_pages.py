"""R3 drafting: page features and seeded job selection (evals/golden/qa/pages.py)."""
from collections import Counter

from evals.chunking.base import SourceDoc, TableSource
from evals.golden.qa.pages import (PageInfo, eligible, is_toc, page_infos, parse_scan_of,
                                   select_jobs)

PROSE = " ".join(["The Borrower shall repay the Loans in full on the Maturity Date."] * 30)


def info(doc="d1_ca", page=1, part="edgar", session="d1", words=300, numbers=0, table_rows=0,
         scan_of=None, **kw):
    return PageInfo(doc, part, session, "original", page, words, numbers, table_rows,
                    scan_of=scan_of, **kw)


# ---- features --------------------------------------------------------------------------------
def test_is_toc():
    toc = "\n".join(f"Section 2.{i} Something Important ........ {i + 10}" for i in range(12))
    assert is_toc(toc)
    assert not is_toc(PROSE)
    assert not is_toc("Section 1.01 ..... 3\nSection 1.02 ..... 4")   # too few lines to call


def test_parse_scan_of():
    assert parse_scan_of("scan_of:cfpb_cd_h25b:p2") == ("cfpb_cd_h25b", 2)
    assert parse_scan_of("original") is None


def test_page_infos_counts_tables_and_flags():
    sig = "By: x\nName: y\nTitle: z\nBy: a\nName: b\nTitle: c " + "word " * 100
    src = SourceDoc("d1_ca", {1: PROSE, 2: "Fee $1,000 and 2.5% due 2025 " * 20, 3: sig},
                    tables=[TableSource(2, 0, "", [["a", "b"], ["1", "2"], ["", ""], ["3", "4"]], [])])
    p1, p2, p3 = page_infos(src, "edgar", "d1", "original")
    assert p1.usable and p1.table_rows == 0 and p1.words == len(PROSE.split())
    assert p2.table_rows == 3                      # blank row not counted
    assert p2.numbers >= 60
    assert p3.signature and not p3.usable


def test_eligible_rules():
    assert eligible(info(), "lookup")
    assert not eligible(info(words=50), "lookup")
    assert not eligible(info(toc=True), "lookup")
    assert eligible(info(table_rows=3), "table") and not eligible(info(table_rows=2), "table")
    assert eligible(info(numbers=6), "calculation") and not eligible(info(numbers=5), "calculation")
    assert eligible(info(), "explain")
    assert not eligible(info(words=150), "explain")
    assert not eligible(info(table_rows=4), "explain")


# ---- selection -------------------------------------------------------------------------------
def corpus():
    out = []
    for s in ("d1", "d2", "d3"):
        out += [info(f"{s}_ca", n, session=s) for n in range(1, 11)]
    out += [info("cfpb_le", n, part="consumer", session="c1", table_rows=5) for n in range(1, 4)]
    out += [PageInfo("scan_01", "simulated_scan", "d1", "scan_of:d1_ca:p1", 1, 200, 0, 0,
                     scan_of=("d1_ca", 1))]
    return out


def test_select_is_deterministic_and_spread():
    plan = {("lookup", "edgar"): 6, ("explain", "edgar"): 3}
    a, short = select_jobs(corpus(), plan)
    b, _ = select_jobs(corpus(), plan)
    assert a == b and not short
    assert Counter(j.session_id for j in a if j.qtype == "lookup") == {"d1": 2, "d2": 2, "d3": 2}
    assert len({j.pages for j in a}) == len(a)     # enough pages: no page shared at all
    assert [j.job_id for j in a][:2] == ["lookup-001", "lookup-002"]


def test_page_once_per_type_shared_across_types_when_short():
    plan = {("table", "consumer"): 3, ("calculation", "consumer"): 3, ("lookup", "consumer"): 1}
    jobs, short = select_jobs(corpus(), plan)
    per_type = Counter((j.qtype, j.pages) for j in jobs)
    assert max(per_type.values()) == 1
    assert not short
    assert len(jobs) == 7


def test_shortfall_reported_and_exhausted_session_skipped():
    infos = [info("d1_ca", 1, session="d1")] + [info("d2_ca", n, session="d2") for n in range(1, 5)]
    jobs, short = select_jobs(infos, {("lookup", "edgar"): 5})
    assert len(jobs) == 5 and not short           # d1 runs out, d2 keeps filling
    jobs, short = select_jobs(infos, {("lookup", "edgar"): 7})
    assert len(jobs) == 5 and short == {"lookup/edgar": 2}


def test_scan_shares_use_with_its_source_page():
    infos = [info("d1_ca", 1, session="d1")] + [
        PageInfo("scan_01", "simulated_scan", "d1", "scan_of:d1_ca:p1", 1, 200, 0, 0, scan_of=("d1_ca", 1))]
    used: dict = {}
    jobs, _ = select_jobs(infos, {("lookup", "scan"): 1, ("lookup", "edgar"): 1}, used=used)
    assert [j.bucket for j in jobs] == ["scan"]   # its source page is taken for lookup
    assert used == {("d1_ca", 1): {"lookup"}}


def test_used_from_pairs_is_respected():
    infos = [info("d1_ca", n, session="d1") for n in (1, 2)]
    jobs, _ = select_jobs(infos, {("lookup", "edgar"): 2}, used={("d1_ca", 1): {"lookup"}})
    assert [j.pages for j in jobs] == [(("d1_ca", 2),)]
