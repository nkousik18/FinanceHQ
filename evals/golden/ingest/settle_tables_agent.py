"""
Settle the golden-set tables where Docling P1 and P2 disagree, plus the from-scratch
tables (S-2026-11-06-8). Done by the agent (Claude) from the page images, at Kousik's
request (2026-11-12); every table it touches is marked `settled_by: "agent"`.

Rules used:
- Grid structure follows the page image. When P1 or P2 already had the right structure,
  start from that grid (`adjudication` = P1 / P2) and fix text that differs from the
  image (OCR errors, text the image doesn't show). When neither had it, type the grid
  from the image (`adjudication` = own).
- Cell text matches the image. Checkbox glyphs and bullets are left out (the metric's
  normalizer drops them anyway).
- From-scratch pages: typed from the image only, no Docling output used.

Run once:  PYTHONPATH=. venv/bin/python -m evals.golden.ingest.settle_tables_agent
"""
from __future__ import annotations

import copy
import json

from evals.golden.ingest import labeling
from evals.golden.ingest.schema import validate_labels
from evals.ingest_metrics.run_s0 import Docs

AGENT = "agent"


# ---------- grid helpers ----------

def grid(rows: list[list], header_rows: list[int] | None = None) -> dict:
    """
    Build a table from rows of cells. A cell is "text", or (text, col_span) or
    (text, col_span, row_span). Positions covered by an earlier span are skipped.
    Rows shorter than the widest row are padded with empty cells.
    """
    occ: set[tuple[int, int]] = set()
    cells = []
    for r, row in enumerate(rows):
        c = 0
        for item in row:
            while (r, c) in occ:
                c += 1
            text, cs, rs = (item, 1, 1) if isinstance(item, str) else (tuple(item) + (1, 1))[:3]
            cells.append({"row": r, "col": c, "row_span": rs, "col_span": cs, "text": text})
            for rr in range(r, r + rs):
                for cc in range(c, c + cs):
                    occ.add((rr, cc))
            c += cs
    n_rows = len(rows)
    n_cols = max(c for _, c in occ) + 1
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in occ:
                cells.append({"row": r, "col": c, "row_span": 1, "col_span": 1, "text": ""})
    cells.sort(key=lambda x: (x["row"], x["col"]))
    return {"num_rows": n_rows, "num_cols": n_cols, "header_rows": header_rows or [], "cells": cells}


def to_rows(t: dict) -> list[list]:
    """Inverse of grid(): rows of (text, col_span, row_span) in anchor order."""
    rows = [[] for _ in range(t["num_rows"])]
    for c in sorted(t["cells"], key=lambda x: (x["row"], x["col"])):
        rows[c["row"]].append((c["text"], c["col_span"], c["row_span"]))
    return rows


def patch(t: dict, text: dict | None = None) -> dict:
    """Copy of a parser table with some cell texts replaced: {(row, col): text}."""
    t = copy.deepcopy(t)
    for c in t["cells"]:
        if (c["row"], c["col"]) in (text or {}):
            c["text"] = text[(c["row"], c["col"])]
    return {k: t[k] for k in ("num_rows", "num_cols", "header_rows", "cells")}


# ---------- pre-filled pages: P1 vs P2 disagreements ----------

def settle_prefilled(P):  # P(page, table_id, cfg) -> parser table matched to that label table
    out = {}

    # Contact table. P1 transposed it. The Mortgage Broker column and the stray "45" exist in
    # the text layer but are not visible on the page, so they are left empty.
    out["cfpb_cd_h25b_p006", "t2"] = ("P2", patch(P("cfpb_cd_h25b_p006", "t2", "P2"), {
        (1, 2): "", (2, 2): "", (3, 2): "", (5, 2): "", (6, 2): "", (8, 2): "", (9, 2): "",
        (2, 3): "789 Local Lane Sometown, ST 12345"}),
        "P2 orientation matches the page; Mortgage Broker column and '45' are invisible text-layer content, blanked")

    # Escrow box: "You may have other property costs." belongs to the Non-Escrowed row (P2);
    # the "Escrow" title spans the whole table (P1 had that right).
    rows = to_rows(P("cfpb_cd_h25e_p005", "t1", "P2"))
    rows[0] = [("Escrow", 3, 1)]
    out["cfpb_cd_h25e_p005", "t1"] = ("P2", grid(rows, [0]), "P2 rows; title row spans 3 columns")

    out["cfpb_cd_h25g_p005", "t1"] = ("P2", patch(P("cfpb_cd_h25g_p005", "t1", "P2"), {
        (2, 2): "Estimated total amount over year 1 for your non-escrowed property costs: You may have other property costs.",
        (3, 2): "A cushion for the escrow account you pay at closing. See Section G on page 2."}),
        "P2 rows; 'You may have other property costs.' moved to the Non-Escrowed row (both parsers misplaced it)")

    out["cfpb_cd_h25e_p004", "t2"] = ("P2", patch(P("cfpb_cd_h25e_p004", "t2", "P2"), {
        (5, 1): "$24,901.00 From To Borrower", (5, 2): "$29,677.43 From To Borrower"}),
        "'Did this change?' is one column (P2); checkbox OCR junk removed")

    out["deal_01_ca_p002", "t1"] = ("P2", patch(P("deal_01_ca_p002", "t1", "P2")),
                                    "P2 keeps one TOC entry per row; P1 merged 4.3/4.4 and 4.6/4.7")

    # Scanned TOC: three columns (P2). OCR misreads of "35" fixed; chapter rows span the
    # number and title columns; last entry wraps onto one row.
    rows = to_rows(patch(P("scan_33_p001", "t1", "P2"), {(2, 2): "35", (6, 2): "35", (8, 2): "35", (17, 2): "43"}))
    rows[13] = [("CONDITIONS TO THE LOANS.", 2, 1), ("40", 1, 1)]
    rows[16] = [("AFFIRMATIVE COVENANTS", 2, 1), ("43", 1, 1)]
    rows[35] = [("Section 6.19", 1, 1), ("Post-Closing Obligations.", 1, 1), ("49", 1, 1)]
    out["scan_33_p001", "t1"] = ("P2", grid(rows[:36]), "P2 columns; OCR fixes (BS/B5 -> 35), wrapped last row joined")

    # Scan of the CD page 1. Closing / Transaction / Loan information: P1 keeps one line per row.
    out["scan_01_p001", "t1"] = ("P1", patch(P("scan_01_p001", "t1", "P1"), {
        (1, 3): "Michael Jones and Mary Stone", (5, 5): "Conventional FHA", (6, 5): "VA"}),
        "P1 keeps one line per row; P2 merged lines. OCR junk removed")

    out["scan_01_p001", "t2"] = ("P1", grid([
        ["Loan Terms", "", "Can this amount increase after closing?"],
        ["Loan Amount", "$162,000", "NO"],
        ["Interest Rate", "3.875%", "NO"],
        ["Monthly Principal & Interest See Projected Payments below for your Estimated Total Monthly Payment", "$761.78", "NO"],
        ["", "", "Does the loan have these features?"],
        ["Prepayment Penalty", "", "YES As high as $3,240 if you pay off the loan during the first 2 years"],
        ["Balloon Payment", "", "NO"],
    ], [0, 4]), "P1's 3 columns (YES and its note are one cell); headers sit over the answer column only; OCR fixes")

    out["scan_01_p001", "t3"] = ("P1", grid([
        ["Payment Calculation", ("Years 1-7", 2), "Years 8-30"],
        ["Principal & Interest", "", "$761.78", "$761.78"],
        ["Mortgage Insurance", "+", "82.35", "+ —"],
        ["Estimated Escrow Amount can increase over time", "+", "206.13", "+ 206.13"],
        ["Estimated Total Monthly Payment", ("$1,050.26", 2), "$967.91"],
        [("Estimated Taxes, Insurance & Assessments Amount can increase over time See page 4 for details", 1, 2),
         ("$356.13 a month", 1, 2),
         ("This estimate includes In escrow? Property Taxes YES Homeowner's Insurance YES "
          "Other: Homeowner's Association Dues NO", 2)],
        [("See Escrow Account on page 4 for details. You must pay for other property costs separately.", 2)],
    ], [0]), "P1's 4-column layout; taxes block rebuilt as merged cells; OCR fixes")

    out["scan_01_p001", "t4"] = ("own", grid([
        ["Closing Costs", "$9,712.10",
         "Includes $4,694.05 in Loan Costs + $5,018.05 in Other Costs - $0 in Lender Credits. See page 2 for details."],
        ["Cash to Close", "$14,147.26",
         "Includes Closing Costs. See Calculating Cash to Close on page 3 for details."],
    ]), "Three columns: P1 put the label last, P2 merged amount and note")

    # Closing Cost Details, h25g p3 Loan Costs: P2 is right except two amounts that sit in At Closing only.
    rows = to_rows(P("cfpb_cd_h25g_p003", "t1", "P2"))
    rows[3] = [rows[3][0], ("$1,500.00", 1, 1), ("", 1, 1), rows[3][-1]]
    rows[4] = [rows[4][0], ("$750.00", 1, 1), ("", 1, 1), rows[4][-1]]
    out["cfpb_cd_h25g_p003", "t1"] = ("P2", grid(rows, [0, 1]), "P2 rows; two At Closing amounts un-spanned")

    # h25g p3 Other Costs: neither parser kept the rows. Columns: item (2 cols wide, to line up
    # with Loan Costs) | Borrower At Closing | Borrower Before Closing | Paid by Others.
    D = lambda s: (s, 2)
    S = lambda s: (s, 2)  # section total spans both Borrower-Paid columns
    out["cfpb_cd_h25g_p003", "t2"] = ("own", grid([
        [D("E. Taxes and Other Government Fees"), S("$80.00"), ""],
        [D("01 Recording Fees Deed: Mortgage: $80.00"), "$80.00", "", ""],
        [D("02"), "", "", ""],
        [D("F. Prepaids"), S("$2,199.15"), ""],
        [D("01 Homeowner's Insurance Premium ( 12 mo.) to Insurance Co."), "$1,209.96", "", ""],
        [D("02 Mortgage Insurance Premium ( 1 mo.) to MI Co."), "$82.35", "", ""],
        [D("03 Prepaid Interest ( $17.19 per day from 4/15/2013 to 5/1/2013 )"), "$275.04", "", ""],
        [D("04 Property Taxes ( 6 mo.) to Any County USA"), "$631.80", "", ""],
        [D("05"), "", "", ""],
        [D("G. Initial Escrow Payment at Closing"), S("$576.95"), ""],
        [D("01 Homeowner's Insurance $100.83 per month for 2 mo."), "$201.66", "", ""],
        [D("02 Mortgage Insurance $82.35 per month for 2 mo."), "$164.70", "", ""],
        [D("03 Property Taxes $105.30 per month for 2 mo."), "$210.60", "", ""],
        [D("04"), "", "", ""], [D("05"), "", "", ""], [D("06"), "", "", ""], [D("07"), "", "", ""],
        [D("08 Aggregate Adjustment"), "- $0.01", "", ""],
        [D("H. Other"), S(""), ""],
        *[[D(f"0{i}"), "", "", ""] for i in range(1, 9)],
        [D("I. TOTAL OTHER COSTS (Borrower-Paid)"), S("$2,856.10"), ""],
        [D("Other Costs Subtotals (E + F + G + H)"), "$2,856.10", "", ""],
        [D("J. TOTAL CLOSING COSTS (Borrower-Paid)"), S("$6,156.60"), ""],
        [D("Closing Costs Subtotals (D + I)"), "$6,216.60", "$440.00", ""],
        [D("Lender Credits"), "- $500.00", "", ""],
    ]), "Neither parser kept the rows; typed from the image")

    # h25b p3: whole Closing Cost Details page as one table, 7 columns:
    # item | payee | Borrower At / Before | Seller At / Before | Paid by Others.
    def line(item, payee="", b_at="", b_before="", s_at="", s_before="", others=""):
        first = [D(item)] if payee is None else [item, payee]
        return first + [b_at, b_before, s_at, s_before, others]

    def section(title, total):
        return [D(title), S(total), "", "", ""]

    def empties(*nums):
        return [line(n, None) for n in nums]

    out["cfpb_cd_h25b_p003", "t1"] = ("own", grid([
        [("Loan Costs", 2, 2), ("Borrower-Paid", 2), ("Seller-Paid", 2), ("Paid by Others", 1, 2)],
        ["At Closing", "Before Closing", "At Closing", "Before Closing"],
        section("A. Origination Charges", "$1,802.00"),
        line("01 0.25 % of Loan Amount (Points)", None, "$405.00"),
        line("02 Application Fee", None, "$300.00"),
        line("03 Underwriting Fee", None, "$1,097.00"),
        *empties("04", "05", "06", "07", "08"),
        section("B. Services Borrower Did Not Shop For", "$236.55"),
        line("01 Appraisal Fee", "to John Smith Appraisers Inc.", others="$405.00"),
        line("02 Credit Report Fee", "to Information Inc.", b_before="$29.80"),
        line("03 Flood Determination Fee", "to Info Co.", "$20.00"),
        line("04 Flood Monitoring Fee", "to Info Co.", "$31.75"),
        line("05 Tax Monitoring Fee", "to Info Co.", "$75.00"),
        line("06 Tax Status Research Fee", "to Info Co.", "$80.00"),
        *empties("07", "08", "09", "10"),
        section("C. Services Borrower Did Shop For", "$2,655.50"),
        line("01 Pest Inspection Fee", "to Pests Co.", "$120.50"),
        line("02 Survey Fee", "to Surveys Co.", "$85.00"),
        line("03 Title - Insurance Binder", "to Epsilon Title Co.", "$650.00"),
        line("04 Title - Lender's Title Insurance", "to Epsilon Title Co.", "$500.00"),
        line("05 Title - Settlement Agent Fee", "to Epsilon Title Co.", "$500.00"),
        line("06 Title - Title Search", "to Epsilon Title Co.", "$800.00"),
        *empties("07", "08"),
        section("D. TOTAL LOAN COSTS (Borrower-Paid)", "$4,694.05"),
        line("Loan Costs Subtotals (A + B + C)", None, "$4,664.25", "$29.80"),
        line("Other Costs", None),
        section("E. Taxes and Other Government Fees", "$85.00"),
        line("01 Recording Fees", "Deed: $40.00 Mortgage: $45.00", "$85.00"),
        line("02 Transfer Tax", "to Any State", s_at="$950.00"),
        section("F. Prepaids", "$2,120.80"),
        line("01 Homeowner's Insurance Premium ( 12 mo.) to Insurance Co.", None, "$1,209.96"),
        line("02 Mortgage Insurance Premium ( mo.)", None),
        line("03 Prepaid Interest ( $17.44 per day from 4/15/13 to 5/1/13 )", None, "$279.04"),
        line("04 Property Taxes ( 6 mo.) to Any County USA", None, "$631.80"),
        *empties("05"),
        section("G. Initial Escrow Payment at Closing", "$412.25"),
        line("01 Homeowner's Insurance $100.83", "per month for 2 mo.", "$201.66"),
        line("02 Mortgage Insurance", "per month for mo."),
        line("03 Property Taxes $105.30", "per month for 2 mo.", "$210.60"),
        *empties("04", "05", "06", "07"),
        line("08 Aggregate Adjustment", None, "- 0.01"),
        section("H. Other", "$2,400.00"),
        line("01 HOA Capital Contribution", "to HOA Acre Inc.", "$500.00"),
        line("02 HOA Processing Fee", "to HOA Acre Inc.", "$150.00"),
        line("03 Home Inspection Fee", "to Engineers Inc.", "$750.00", s_before="$750.00"),
        line("04 Home Warranty Fee", "to XYZ Warranty Inc.", s_at="$450.00"),
        line("05 Real Estate Commission", "to Alpha Real Estate Broker", s_at="$5,700.00"),
        line("06 Real Estate Commission", "to Omega Real Estate Broker", s_at="$5,700.00"),
        line("07 Title - Owner's Title Insurance (optional)", "to Epsilon Title Co.", "$1,000.00"),
        *empties("08"),
        section("I. TOTAL OTHER COSTS (Borrower-Paid)", "$5,018.05"),
        line("Other Costs Subtotals (E + F + G + H)", None, "$5,018.05"),
        section("J. TOTAL CLOSING COSTS (Borrower-Paid)", "$9,712.10"),
        line("Closing Costs Subtotals (D + I)", None, "$9,682.30", "$29.80", "$12,800.00", "$750.00", "$405.00"),
        line("Lender Credits", None),
    ], [0, 1]), "Both parsers scrambled the rows; typed from the image")
    return out


# ---------- from-scratch pages ----------

def two_col(rows):
    return grid([list(r) for r in rows], [0])


LE_P3 = {  # element id -> table (Loan Estimate page 2, Closing Cost Details)
    "e2": two_col([("A. Origination Charges", "$1,802"), (".25 % of Loan Amount (Points)", "$405"),
                   ("Application Fee", "$300"), ("Underwriting Fee", "$1,097")]),
    "e3": two_col([("E. Taxes and Other Government Fees", "$85"), ("Recording Fees and Other Taxes", "$85"),
                   ("Transfer Taxes", ""), ("F. Prepaids", "$867"),
                   ("Homeowner's Insurance Premium ( 6 months)", "$605"), ("Mortgage Insurance Premium ( months)", ""),
                   ("Prepaid Interest ( $17.44 per day for 15 days @ 3.875%)", "$262"), ("Property Taxes ( months)", "")]),
    "e4": two_col([("G. Initial Escrow Payment at Closing", "$413"),
                   ("Homeowner's Insurance $100.83 per month for 2 mo.", "$202"),
                   ("Mortgage Insurance per month for mo.", ""),
                   ("Property Taxes $105.30 per month for 2 mo.", "$211")]),
    "e5": two_col([("B. Services You Cannot Shop For", "$672"), ("Appraisal Fee", "$405"), ("Credit Report Fee", "$30"),
                   ("Flood Determination Fee", "$20"), ("Flood Monitoring Fee", "$32"), ("Tax Monitoring Fee", "$75"),
                   ("Tax Status Research Fee", "$110")]),
    "e6": two_col([("H. Other", "$1,017"), ("Title - Owner's Title Policy (optional)", "$1,017")]),
    "e7": two_col([("C. Services You Can Shop For", "$3,198"), ("Pest Inspection Fee", "$135"), ("Survey Fee", "$65"),
                   ("Title - Insurance Binder", "$700"), ("Title - Lender's Title Policy", "$535"),
                   ("Title - Settlement Agent Fee", "$502"), ("Title - Title Search", "$1,261")]),
    "e8": two_col([("J. TOTAL CLOSING COSTS", "$8,054"), ("D + I", "$8,054"), ("Lender Credits", "")]),
    "e9": two_col([("I. TOTAL OTHER COSTS (E + F + G + H)", "$2,382")]),
    "e10": grid([["Total Closing Costs (J)", "$8,054"],
                 ["Closing Costs Financed (Paid from your Loan Amount)", "$0"],
                 ["Down Payment/Funds from Borrower", "$18,000"], ["Deposit", "- $10,000"],
                 ["Funds for Borrower", "$0"], ["Seller Credits", "$0"],
                 ["Adjustments and Other Credits", "$0"], ["Estimated Cash to Close", "$16,054"]]),
    "e11": two_col([("D. TOTAL LOAN COSTS (A + B + C)", "$5,672")]),
}

SCAN_11 = grid([
    ["Comparisons", ("Use these measures to compare this loan with other loans.", 2)],
    [("In 5 Years", 1, 2), "$54,944", "Total you will have paid in principal, interest, mortgage insurance, and loan costs."],
    ["$0", "Principal you will have paid off."],
    ["Annual Percentage Rate (APR)", "4.617%", "Your costs over the loan term expressed as a rate. This is not your interest rate."],
    ["Total Interest Percentage (TIP)", "81.18%",
     "The total amount of interest that you will pay over the loan term as a percentage of your loan amount."],
], [0])

DEAL_10 = grid([
    ["", "Debt Rating", "Euro-Rate Spread", "Base Rate Spread", "Commitment Fee",
     "Standby Letter of Credit Fee", "Commercial Letter of Credit Fee"],
    ["Tier I", "A-/A3 or higher", "100.0 bps", "0.0 bps", "10.0 bps", "100.0 bps", "50.00 bps"],
    ["Tier II", "BBB+/Baa1", "112.5 bps", "12.5 bps", "12.5 bps", "112.5 bps", "56.25 bps"],
    ["Tier III", "BBB/ Baa2", "125.0 bps", "25.0 bps", "15.0 bps", "125.0 bps", "62.50 bps"],
    ["Tier IV", "BBB-/Baa3", "137.5 bps", "37.5 bps", "20.0 bps", "137.5 bps", "68.75 bps"],
    ["Tier V", "BB+/ Ba1", "150.0 bps", "50.0 bps", "25.0 bps", "150.0 bps", "75.00 bps"],
    ["Tier VI", "<BB+/ Ba1", "175.0 bps", "75.0 bps", "30.0 bps", "175.0 bps", "87.50 bps"],
], [0])


# ---------- apply ----------

def _load(page_id):
    return json.loads((labeling.PAGES / f"{page_id}.json").read_text())


def _save(lab):
    errs = validate_labels(lab)
    if errs:
        raise SystemExit(f"{lab['page_id']}: {errs}")
    (labeling.PAGES / f"{lab['page_id']}.json").write_text(json.dumps(lab, indent=2, ensure_ascii=False))


def _note(lab, text):
    lab["notes"] = (lab.get("notes") or "").strip()
    lab["notes"] = (lab["notes"] + " " if lab["notes"] else "") + text


def main() -> int:
    from evals.golden.ingest.app.compare import compare_page
    docs = Docs()
    cache = {}

    def P(page_id, table_id, cfg):
        if page_id not in cache:
            lab = _load(page_id)
            cache[page_id] = compare_page(lab, {c: docs.view(c, lab["doc_id"], lab["page"]) for c in ("P1", "P2")})
        t = next(t for t in cache[page_id]["tables"] if t["table_id"] == table_id)
        return t[cfg]

    settled = settle_prefilled(P)
    for page_id in sorted({p for p, _ in settled}):
        lab = _load(page_id)
        for t in lab["tables"]:
            if (page_id, t["id"]) in settled:
                adj, g, why = settled[page_id, t["id"]]
                t.update(g)
                t.update({"adjudication": adj, "settled_by": AGENT, "settle_note": why})
        _note(lab, "Disputed P1/P2 tables settled by the agent from the page image (2026-11-12, settle_tables_agent.py).")
        _save(lab)
        print("settled", page_id, [t for p, t in settled if p == page_id])

    # Loan Estimate p3: the one table entry sat on the page heading (e1); the 10 tables drawn
    # on the page had no grids. e1 becomes a section header; e2..e11 get typed grids.
    lab = _load("cfpb_le_h24b_p003")
    for e in lab["layout_elements"]:
        if e["id"] == "e1":
            e["class"] = "section_header"
    by_el = {e["id"]: e for e in lab["layout_elements"]}
    lab["tables"] = [{"id": f"t{i}", "element_id": eid, "bbox": by_el[eid]["bbox"], **g, "settled_by": AGENT}
                     for i, (eid, g) in enumerate(LE_P3.items(), 1)]
    _note(lab, "Agent (2026-11-12): e1 was a heading, not a table; typed grids for the 10 tables e2-e11 from the image.")
    _save(lab)
    print("typed cfpb_le_h24b_p003", len(lab["tables"]), "tables")

    for page_id, g in (("scan_11_p001", SCAN_11), ("deal_10_ca_p133", DEAL_10)):
        lab = _load(page_id)
        assert len(lab["tables"]) == 1
        lab["tables"][0].update(g)
        lab["tables"][0]["settled_by"] = AGENT
        _note(lab, "Agent (2026-11-12): table grid typed from the image.")
        _save(lab)
        print("typed", page_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
