"""R3 drafting: amendment ↔ original and LE ↔ CD pairs (evals/golden/qa/pairs.py)."""
from evals.golden.qa.pages import PageInfo
from evals.golden.qa.pairs import (Pair, Target, amended_targets, amendment_pairs, find_definition,
                                   find_section, form_pairs, is_contents, select_pair_jobs)

AMEND_P1 = '''2. Amendment. The Credit Agreement is hereby amended as follows:
(a) Section 1.01 of the Credit Agreement is hereby amended by amending and restating the definition of "Loan Documents" to read as follows:
"Loan Documents" shall mean this Agreement, the Amendment No. 1 and any Note.
"Applicable Margin" means 1.25% per annum.
(c) Section 5.03 of the Credit Agreement is hereby amended and restated in its entirety to read as follows:
SECTION 5.03 Financial Covenant. The Leverage Ratio shall not exceed 3.75 to 1.00.
1.1. Clause (a) of Section 5.08 of the Credit Agreement is amended and restated in its entirety to read as follows:
(d) The definition of "Maturity Date" in Section 1.01 of the Credit Agreement is hereby amended and restated as follows:'''

ORIG = {
    2: "\n".join(f"SECTION {a}.{b:02d} Some Heading Number {a}{b}" for a in range(1, 6) for b in range(1, 5)),
    20: 'Lorem.\n"Loan Documents" shall mean this Agreement and any Note.\n"Maturity Date" means June 30, 2024.',
    21: '"Applicable Margin" means 1.50% per annum.',
    62: "Prior text.\nSECTION 5.03. Financial Covenant. So long as any Advance remains unpaid, 3.50 to 1.00.",
    70: "5.08 Use of Proceeds. The proceeds will be used for working capital.",
}


def test_amended_targets_sections_and_definitions_in_order():
    ts = amended_targets(AMEND_P1)
    assert ts == [Target("definition", "Loan Documents"), Target("definition", "Applicable Margin"),
                  Target("section", "5.03"), Target("section", "5.08"), Target("definition", "Maturity Date")]


def test_definitions_section_terms_stop_at_next_amended_section():
    text = ('Section 1.1 of the Credit Agreement is hereby amended by adding:\n"Alpha" means a.\n'
            'Section 7.9 of the Credit Agreement is hereby amended as follows:\n"Beta" means b.')
    assert amended_targets(text) == [Target("definition", "Alpha"), Target("section", "7.9")]


def test_is_contents():
    assert is_contents(ORIG[2])            # 20 distinct section headings
    assert not is_contents(ORIG[62])
    doubled = "7.10 Use of funds. x 7.11 Financial Covenants. y " * 2   # image-filing double text
    assert not is_contents(doubled)


def test_find_section_skips_toc_and_returns_title():
    assert find_section(ORIG, "5.03") == (62, "Financial Covenant")
    assert find_section(ORIG, "5.08") == (70, "Use of Proceeds")
    assert find_section(ORIG, "9.99") is None
    toc_line = {1: "SECTION 6.01 Indebtedness 88\nother", 5: "SECTION 6.01 Indebtedness. The Borrower will not"}
    assert find_section(toc_line, "6.01") == (5, "Indebtedness")


def test_find_definition():
    assert find_definition(ORIG, "Loan Documents") == 20
    assert find_definition(ORIG, "Applicable Margin") == 21
    assert find_definition(ORIG, "Nope") is None


def test_amendment_pairs_and_money_flag():
    texts = {"d_am1": {1: AMEND_P1}, "d_ca": ORIG}
    roles = {"d_am1": ("d", "amendment_1"), "d_ca": ("d", "original")}
    pairs = amendment_pairs(texts, roles)
    got = {(p.orig[1], p.label) for p in pairs}
    assert got == {(20, 'definition "Loan Documents"'), (21, 'definition "Applicable Margin"'),
                   (62, "Section 5.03 Financial Covenant"), (70, "Section 5.08 Use of Proceeds"),
                   (20, 'definition "Maturity Date"')}
    money = {p.label for p in pairs if p.money}
    assert money == {'definition "Applicable Margin"', "Section 5.03 Financial Covenant", 'definition "Maturity Date"'}
    assert not Pair("d", ("a", 1), ("b", 1), "Section 11.16 Counterparts; Integration").money


def _page(doc, n, session, words=200):
    return PageInfo(doc, "consumer", session, "form", n, words, 10, 5)


def test_form_pairs_same_page_within_session():
    infos = [_page(d, n, "s1") for d in ("cfpb_le_a", "cfpb_cd_a") for n in (1, 2, 3, 4)]
    infos += [_page("cfpb_le_b", 2, "s2")]                    # no CD in s2
    infos[1] = _page("cfpb_le_a", 2, "s1", words=10)          # unusable page
    pairs = form_pairs(infos)
    assert [(p.amend, p.orig) for p in pairs] == [(("cfpb_le_a", 3), ("cfpb_cd_a", 3))]


def test_select_pair_jobs_money_first_spread_and_marks_used():
    am = [Pair("d1", ("d1_am1", 1), ("d1_ca", 5), "Section 9.1 Notices"),
          Pair("d1", ("d1_am1", 2), ("d1_ca", 6), 'definition "Applicable Rate"'),
          Pair("d2", ("d2_am1", 1), ("d2_ca", 9), "Section 2.10 Fees"),
          Pair("d2", ("d2_am1", 1), ("d2_ca", 10), 'definition "Maturity Date"')]   # same amend page
    forms = [Pair("c1", ("cfpb_le", 2), ("cfpb_cd", 2), "LE vs CD")]
    used: dict = {}
    jobs, short = select_pair_jobs(am, forms, used=used, n_amend=3, n_form=2)
    d2_money = {"Section 2.10 Fees", 'definition "Maturity Date"'}
    assert jobs[0].note in d2_money | {'definition "Applicable Rate"'}
    assert {jobs[0].session_id, jobs[1].session_id} == {"d1", "d2"}       # money pairs, one per deal
    assert jobs[2].note == "Section 9.1 Notices"             # d2's other pair reuses its amend page
    assert all(j.amendment and j.qtype == "cross_doc" for j in jobs[:3])
    assert not jobs[3].amendment and jobs[3].bucket == "consumer"
    assert short == {"cross_doc/form": 1}
    assert used[("d1_ca", 6)] == {"cross_doc"}
    assert sum(("d2_ca", n) in used for n in (9, 10)) == 1
    again, _ = select_pair_jobs(am, forms, n_amend=3, n_form=2)
    assert again == jobs
