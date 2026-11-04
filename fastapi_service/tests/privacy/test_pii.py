"""Tests for app/privacy/pii.py (placeholder PII module, not wired into the pipeline)."""
import pytest

from app.privacy.pii import (
    PIIType,
    RedactionMode,
    aba_valid,
    detect,
    luhn_valid,
    redact,
)


def _types(text):
    return [m.type for m in detect(text)]


class TestChecksums:
    def test_luhn_accepts_valid_test_card(self):
        assert luhn_valid("4111 1111 1111 1111")

    def test_luhn_rejects_off_by_one(self):
        assert not luhn_valid("4111 1111 1111 1112")

    def test_aba_accepts_valid_routing_number(self):
        assert aba_valid("021000021")

    def test_aba_rejects_invalid_routing_number(self):
        assert not aba_valid("021000022")


class TestDetect:
    @pytest.mark.parametrize("text,expected", [
        ("SSN: 123-45-6789", PIIType.SSN),
        ("Employer ID 12-3456789", PIIType.EIN),
        ("Card 4111-1111-1111-1111 on file", PIIType.CREDIT_CARD),
        ("Email jane.doe@example.com today", PIIType.EMAIL),
        ("Call (555) 123-4567", PIIType.PHONE),
        ("PAN ABCDE1234F", PIIType.PAN),
        ("Routing number: 021000021", PIIType.ROUTING_NUMBER),
        ("Account No: 000123456789", PIIType.BANK_ACCOUNT),
    ])
    def test_detects_each_type(self, text, expected):
        assert expected in _types(text)

    @pytest.mark.parametrize("text", [
        "Loan amount $1,250,000,000 at 7.25% per annum",
        "Applicable Margin: 1.75% for SOFR Loans; 0.75% for Base Rate Loans",
        "Section 2.11(b)(iii) Mandatory Prepayments",
        "Maturity Date: 03/15/2029",
        "Aggregate Commitments of 750000000",
        "Invalid SSN 000-12-3456 and 666-12-3456",
    ])
    def test_no_false_positives_on_loan_figures(self, text):
        assert detect(text) == []

    def test_unlabelled_nine_digits_not_routing(self):
        assert detect("Reference 021000021") == []

    def test_invalid_card_checksum_ignored(self):
        assert PIIType.CREDIT_CARD not in _types("Card 4111 1111 1111 1112")

    def test_matches_sorted_and_non_overlapping(self):
        matches = detect("a@b.co then 123-45-6789 then (555) 123-4567")
        starts = [m.start for m in matches]
        assert starts == sorted(starts)
        for a, b in zip(matches, matches[1:]):
            assert a.end <= b.start


class TestRedact:
    TEXT = "Borrower SSN 123-45-6789, email jane@example.com."

    def test_mask_mode(self):
        r = redact(self.TEXT)
        assert r.text == "Borrower SSN [SSN], email [EMAIL]."
        assert r.counts == {"ssn": 1, "email": 1}

    def test_partial_mode_keeps_last_four(self):
        r = redact("SSN 123-45-6789", mode=RedactionMode.PARTIAL)
        assert r.text == "SSN *******6789"

    def test_hash_mode_is_deterministic_and_salted(self):
        a = redact(self.TEXT, mode=RedactionMode.HASH, salt="s1").text
        b = redact(self.TEXT, mode=RedactionMode.HASH, salt="s1").text
        c = redact(self.TEXT, mode=RedactionMode.HASH, salt="s2").text
        assert a == b
        assert a != c
        assert "123-45-6789" not in a

    def test_labelled_account_keeps_label(self):
        r = redact("Account No: 000123456789")
        assert r.text == "Account No: [BANK_ACCOUNT]"

    def test_no_pii_returns_text_unchanged(self):
        text = "The Applicable Margin is 2.00%."
        r = redact(text)
        assert r.text == text
        assert r.matches == []
