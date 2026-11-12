"""F8 tests for text normalization and text accuracy metrics, on inputs with known answers."""
import pytest

from evals.ingest_metrics.normalize import norm_match, norm_text, numeric_tokens
from evals.ingest_metrics.text import bag_word_recall, cer, numeric_fidelity, text_layer_usable, wer


class TestNormalize:
    def test_quotes_dashes_spaces(self):
        assert norm_text("“Loan” — Borrower’s   fee ") == "\"Loan\" - Borrower's fee"

    def test_case_kept_for_text_but_not_for_match(self):
        assert norm_text("Loan Amount:") == "Loan Amount:"
        assert norm_match("Loan Amount:") == "loan amount"

    @pytest.mark.parametrize("s,expected", [
        ("Loan Amount $162,000", ["162000"]),
        ("rate of 3.875% and 2.00 percent", ["3.875%", "2.00%"]),
        ("Indian grouping 7,50,000", ["750000"]),
        ("a credit of (1,000) applies", ["-1000"]),
        ("due 03/15/2029 or 3/1/24", ["03/15/2029", "3/1/24"]),
        ("Section 2.11 and 1.01", ["2.11", "1.01"]),
        ("Form H24B sample 10b", []),
        ("Sections 2.01-2.05 and years 2019-2020", ["2.01", "2.05", "2019", "2020"]),
        ("Deposit - $10,000 and -$500", ["10000", "-500"]),
    ])
    def test_numeric_tokens(self, s, expected):
        assert numeric_tokens(s) == expected


class TestTextMetrics:
    def test_cer_known_value(self):
        assert cer("kitten", "sitting") == pytest.approx(3 / 7)   # 3 edits, 7 reference chars

    def test_cer_identical_after_normalization(self):
        assert cer("Loan  “fee”", 'Loan "fee"') == 0.0

    def test_wer_known_value(self):
        assert wer("the loan is due", "the loan was due today") == pytest.approx(2 / 5)

    def test_empty_reference_is_none(self):
        assert cer("x", "") is None and wer("x", " ") is None

    def test_numeric_fidelity_multiset_and_order_free(self):
        ref = "Fee $300, Fee $300, rate 3.875%"
        assert numeric_fidelity("rate 3.875% Fee $300", ref) == pytest.approx(2 / 3)   # one $300 missing
        assert numeric_fidelity("3.875% $300 $300", ref) == 1.0                        # order irrelevant
        assert numeric_fidelity("anything", "no numbers here") is None

    def test_ocr_digit_error_counts(self):
        assert numeric_fidelity("Loan Amount $162,O00", "Loan Amount $162,000") == 0.0


def test_text_layer_usable():
    assert text_layer_usable("word " * 60) == (True, "")
    ok, why = text_layer_usable("too short")
    assert not ok and "words" in why
    ok, why = text_layer_usable(("word " * 60) + "\x00" * 50)
    assert not ok and "non-printable" in why


def test_bag_word_recall_ignores_order_but_cer_does_not():
    ref, reordered = "Loan Amount 162,000 Loan Term 30 years", "Loan Term 30 years Loan Amount 162,000"
    assert bag_word_recall(reordered, ref) == 1.0
    assert cer(reordered, ref) > 0.3
    assert bag_word_recall("Loan Amount", ref) == pytest.approx(2 / 7)
