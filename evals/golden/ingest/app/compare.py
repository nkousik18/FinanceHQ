"""
P1 vs P2 table comparison for adjudicating the golden set (S-2026-11-06-7).

Why: the pre-filled table labels were Docling P1's output and verification left the
cell text unchanged, so S0 cell accuracy favoured P1 by construction. For every
labelled table where P1 and P2 read differently, the labeller looks at the image next
to both grids and picks or fixes the correct one. The choice is stored on the label
table as `adjudication` (P1 / P2 / own) so the S0 report can say how many tables went
each way.

Highlights compare P1 with P2 (not with the label), so they don't change while editing.
"""
from __future__ import annotations

from evals.ingest_metrics.normalize import norm_match
from evals.ingest_metrics.tables import cell_accuracy_counts, match_tables

CONFIGS = ("P1", "P2")


def _anchor(table: dict, r: int, c: int) -> dict | None:
    for cell in table["cells"]:
        if cell["row"] <= r < cell["row"] + cell["row_span"] and cell["col"] <= c < cell["col"] + cell["col_span"]:
            return cell
    return None


def differing_cells(a: dict | None, b: dict | None) -> list[list[int]]:
    """Anchor (row, col) of cells in `a` that differ from `b` in text or span. All of them if b is None."""
    if a is None:
        return []
    out = []
    for cell in a["cells"]:
        other = None if b is None else _anchor(b, cell["row"], cell["col"])
        same = (other is not None and other["row"] == cell["row"] and other["col"] == cell["col"]
                and other["row_span"] == cell["row_span"] and other["col_span"] == cell["col_span"]
                and norm_match(other["text"]) == norm_match(cell["text"]))
        if not same:
            out.append([cell["row"], cell["col"]])
    return out


def _acc(label: dict, pred: dict | None) -> float | None:
    c, t = cell_accuracy_counts(label, pred)
    return c / t if t else None


def compare_page(lab: dict, views: dict) -> dict:
    """
    `views` maps config -> PageView (or None). Returns, per label table, the matched
    P1 and P2 tables, which cells differ between them, and whether they disagree.
    """
    preds = {cfg: (views[cfg].tables if views.get(cfg) is not None else []) for cfg in CONFIGS}
    matched = {cfg: {li: preds[cfg][pi] for li, pi, _ in match_tables(lab["tables"], preds[cfg])} for cfg in CONFIGS}
    out = []
    for i, t in enumerate(lab["tables"]):
        p1, p2 = matched["P1"].get(i), matched["P2"].get(i)
        d1, d2 = differing_cells(p1, p2), differing_cells(p2, p1)
        disagree = bool(d1 or d2) or (p1 is None) != (p2 is None)
        out.append({
            "table_id": t["id"], "element_id": t["element_id"], "disagree": disagree,
            "adjudication": t.get("adjudication"),
            "label_acc": {"P1": _acc(t, p1), "P2": _acc(t, p2)},
            "P1": p1, "P2": p2, "P1_diff": d1, "P2_diff": d2,
        })
    return {"page_id": lab["page_id"], "from_scratch": lab["from_scratch"], "tables": out}


def review_state(cmp: dict) -> str | None:
    """None = nothing to adjudicate; 'todo' = some disagreeing table not yet decided; 'done'."""
    if cmp["from_scratch"]:
        return None
    todo = [t for t in cmp["tables"] if t["disagree"]]
    if not todo:
        return None
    return "done" if all(t["adjudication"] for t in todo) else "todo"
