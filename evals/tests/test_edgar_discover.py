"""Unit tests for the pure matching logic in evals/corpus/edgar_discover.py (no network)."""
from datetime import datetime

from evals.corpus.edgar_discover import (
    amended_agreement_date,
    amendment_ordinal,
    dedupe_amendments,
    html_to_text,
    mentions_date,
)


class TestAmendedAgreementDate:
    def test_finds_original_date(self):
        text = ("FIRST AMENDMENT TO CREDIT AGREEMENT (this Amendment), dated as of May 12, 2025, "
                "amends that certain Credit Agreement, dated as of January 6, 2025, among ...")
        assert amended_agreement_date(text) == datetime(2025, 1, 6)

    def test_skips_amendment_own_title_date(self):
        text = ("AMENDMENT NO. 2 TO CREDIT AGREEMENT dated as of October 20, 2023 ... reference is "
                "made to the Credit Agreement dated as of October 18, 2021")
        assert amended_agreement_date(text) == datetime(2021, 10, 18)

    def test_none_when_absent(self):
        assert amended_agreement_date("no dates here") is None


class TestOrdinal:
    def test_word_and_number_forms_match(self):
        assert amendment_ordinal("Exhibit 10.1 SECOND AMENDMENT TO CREDIT AGREEMENT") == "2"
        assert amendment_ordinal("EXECUTION VERSION AMENDMENT NO. 2 TO CREDIT AGREEMENT") == "2"

    def test_file_name_does_not_confuse_lookup(self):
        assert amendment_ordinal("EX-10.1 2 secondamendmenttocredi.htm EXHIBIT 10.1 SECOND AMENDMENT TO") == "2"

    def test_title_after_redaction_notice(self):
        notice = "CERTAIN INFORMATION IN THIS DOCUMENT, MARKED BY [***], HAS BEEN EXCLUDED " * 4
        assert amendment_ordinal(notice + "FIRST AMENDMENT TO CREDIT AGREEMENT") == "1"


def _a(date, opening, chars, name="x.htm"):
    return {"file_date": date, "opening": opening, "chars": chars, "filename": name}


class TestDedupe:
    def test_same_doc_filed_twice_is_deduped_even_if_lengths_differ(self):
        own = "SECOND AMENDMENT TO CREDIT AGREEMENT is entered into as of June 24, 2020"
        out = dedupe_amendments([
            _a("2020-07-27", own, 24080),
            _a("2020-06-26", own, 19528),
            _a("2021-02-22", "THIRD AMENDMENT TO CREDIT AGREEMENT", 9000),
        ])
        assert [a["file_date"] for a in out] == ["2020-07-27", "2021-02-22"]   # longer copy kept

    def test_two_different_docs_same_ordinal_is_ambiguous(self):
        assert dedupe_amendments([
            _a("2019-02-06", "FIRST AMENDMENT TO CREDIT AGREEMENT", 20000),
            _a("2019-02-06", "FIRST AMENDMENT TO CREDIT AGREEMENT", 45000),
        ]) is None

    def test_unredacted_copy_preferred(self):
        own = "SECOND AMENDMENT dated as of December 13, 2023"
        redacted = {**_a("2024-03-20", own, 9675), "redactions": 2}
        full = {**_a("2025-03-20", own, 9660), "redactions": 0}
        assert dedupe_amendments([redacted, full])[0]["file_date"] == "2025-03-20"

    def test_same_ordinal_different_dates_is_ambiguous(self):
        assert dedupe_amendments([
            _a("2019-02-06", "FIRST AMENDMENT dated as of January 31, 2019", 20000),
            _a("2019-02-07", "FIRST AMENDMENT dated as of February 1, 2019", 20000),
        ]) is None


def test_mentions_date_handles_spacing_and_padding():
    d = datetime(2021, 3, 4)
    assert mentions_date("dated as of March  4,2021 among", d)
    assert mentions_date("dated as of March 04, 2021", d)
    assert not mentions_date("dated as of March 5, 2021", d)


def test_html_to_text_drops_scripts_and_collapses_space():
    raw = b"<html><style>p{}</style><p>CREDIT\n\n AGREEMENT</p><script>x()</script></html>"
    assert html_to_text(raw) == "CREDIT AGREEMENT"
