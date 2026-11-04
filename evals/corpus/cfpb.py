"""
CFPB consumer loan forms (scope v2 §5.0): the completed English sample
Loan Estimates (H-24) and Closing Disclosures (H-25) published at
https://www.consumerfinance.gov/compliance/compliance-resources/mortgage-resources/tila-respa-integrated-disclosures/forms-samples/

consumerfinance.gov blocks scripted downloads (HTTP 403), and we don't work
around bot protection. So the PDFs are downloaded once by hand into
data/corpus/inbox/cfpb/ and imported from there; the manifest keeps the
official URL and the file's sha256, so any copy can be verified.

US government works: public domain.
"""
from __future__ import annotations

from pathlib import Path

from evals.corpus.manifest import DocRow

REPO = Path(__file__).resolve().parents[2]
INBOX = REPO / "data" / "corpus" / "inbox" / "cfpb"
BASE = "https://files.consumerfinance.gov/f/"

# (doc_id, session, form code, title, file name). A session groups forms for
# the same example loan, confirmed from their contents on 2026-11-04:
#   purchase  = 456 Somewhere Ave, $162,000 (H-24(B), H-25(B))
#   refinance = 123 Anywhere Street, $150,000 (H-24(D), H-25(E)/(F)/(G))
# H-24(C)/(E)/(F) reuse the purchase borrower and address with different loan
# amounts, and H-25(C)/(D) are stand-alone page-3 excerpts, so each is its own
# session (grouping them would put contradicting figures in one session).
FORMS = [
    ("cfpb_le_h24b", "cfpb_purchase_fixed", "H-24(B)", "Loan Estimate – fixed rate loan",
     "201403_cfpb_loan-estimate_fixed-rate-loan-sample-H24B.pdf"),
    ("cfpb_cd_h25b", "cfpb_purchase_fixed", "H-25(B)", "Closing Disclosure – fixed rate loan",
     "201403_cfpb_closing-disclosure_cover-H25B.pdf"),
    ("cfpb_cd_h25c", "cfpb_cd_second_lien", "H-25(C)", "Closing Disclosure p.3 – simultaneous second-lien funds",
     "201403_cfpb_closing-disclosure_cover-H25C.pdf"),
    ("cfpb_cd_h25d1", "cfpb_cd_outside_closing", "H-25(D)", "Closing Disclosure p.3 – funds paid outside of closing",
     "201403_cfpb_closing-disclosure_cover-H25D1.pdf"),
    ("cfpb_le_h24d", "cfpb_refinance", "H-24(D)", "Loan Estimate – refinance",
     "201403_cfpb_loan-estimate_refinance-sample-H24D.pdf"),
    ("cfpb_cd_h25e", "cfpb_refinance", "H-25(E)", "Closing Disclosure – refinance",
     "201403_cfpb_closing-disclosure_cover-H25E.pdf"),
    ("cfpb_cd_h25f1", "cfpb_refinance", "H-25(F)", "Closing Disclosure – refinance, costs over good-faith limits",
     "201403_cfpb_closing-disclosure_cover-H25F1.pdf"),
    ("cfpb_cd_h25g", "cfpb_refinance", "H-25(G)", "Closing Disclosure – refinance, consumer pays extra funds",
     "201403_cfpb_closing-disclosure_cover-H25G.pdf"),
    ("cfpb_le_h24c", "cfpb_le_io_arm", "H-24(C)", "Loan Estimate – interest-only adjustable rate",
     "201403_cfpb_loan-estimate_interest-only-adjustable-rate-loan-sample-H24C.pdf"),
    ("cfpb_le_h24e", "cfpb_le_balloon", "H-24(E)", "Loan Estimate – balloon payment",
     "201403_cfpb_loan-estimate_baloon-payment-H24E.pdf"),
    ("cfpb_le_h24f", "cfpb_le_negam", "H-24(F)", "Loan Estimate – negative amortization",
     "201403_cfpb_loan-estimate_negative-amortization-sample-H24F.pdf"),
]


def consumer_rows() -> list[DocRow]:
    return [
        DocRow(doc_id=doc_id, part="consumer", deal_id=session, role="form",
               company=f"CFPB sample {code}: {title}", form=code, source_url=BASE + fname,
               source_format="pdf", filing_date="2014-02")
        for doc_id, session, code, title, fname in FORMS
    ]


def inbox_path(row: DocRow) -> Path:
    return INBOX / row.source_url.rsplit("/", 1)[1]


def missing_files() -> list[DocRow]:
    return [r for r in consumer_rows() if not inbox_path(r).exists()]
