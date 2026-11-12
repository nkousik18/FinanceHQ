"""
Render the S0 results tables (markdown) from data/results/s0/summary.json, so the
numbers in docs/20_s0_results.md are generated, not typed (F7).

    venv/bin/python -m evals.ingest_metrics.s0_report [summary.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT = Path("data/results/s0/summary.json")
CLEANING = Path("data/results/s0/cleaning.json")
SUBSETS = ["all", "prefilled", "from_scratch", "stratum:digital_prose", "stratum:table_heavy",
           "stratum:complex_layout", "stratum:scanned"]


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def _ci(m):
    if not m or m.get("point") is None:
        return "–"
    return f"{_pct(m['point'])} ({_pct(m['lo'])}–{_pct(m['hi'])}), n={m['n']}"


def _test(t):
    return (f"{_pct(t['diff'])} pts (95% CI {_pct(t['lo'])} to {_pct(t['hi'])}), p={t['p']:.3f}, "
            f"n={t['n']} tables → **{t['winner']}** ({t['rule']})")


def render(s: dict) -> str:
    g, c = s["golden"], s["corpus"]
    cfgs = list(g["configs"])
    out = [f"Run: {g['verified']}/{g['labels']} verified pages, configs {', '.join(cfgs)}, dry_run={s['dry_run']}.", ""]

    out += ["### Decision (D3) and H10", "",
            f"- **Decision, P2 − P1 cell accuracy:** {_test(g['decision'])}",
            f"- **H10, table-heavy stratum:** {_test(g['h10'])}. Verdict: **{g['h10']['verdict']}**", ""]

    out += ["### ★ Golden-set metrics (% , 95% bootstrap CI over pages/tables)", ""]
    for metric, label in (("cell_accuracy", "Cell accuracy"), ("cer_median", "CER median"),
                          ("numeric_fidelity", "Numeric fidelity"), ("section_header_recall", "Section-header recall")):
        out += [f"**{label}**", "", "| Subset | pages | " + " | ".join(cfgs) + " |",
                "|---|---|" + "---|" * len(cfgs)]
        for sub in SUBSETS:
            if sub not in g["configs"][cfgs[0]]:
                continue
            pages = g["configs"][cfgs[0]][sub]["pages"]
            vals = []
            for cfg in cfgs:
                m = g["configs"][cfg][sub]
                v = m.get(metric)
                if metric == "section_header_recall":
                    v = (m.get("class_f1") or {}).get("section_header")
                    v = None if v is None else {"point": v["recall"], "lo": None, "hi": None, "n": v["n_label"]}
                    vals.append("–" if v is None or v["point"] is None else f"{_pct(v['point'])} (n={v['n']})")
                else:
                    vals.append(_ci(v))
            out.append(f"| {sub} | {pages} | " + " | ".join(vals) + " |")
        out.append("")

    out += ["**Diagnostics (all pages):**", "", "| | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
    for key, lab in (("teds_mean", "TEDS mean"), ("header_row_accuracy", "Header-row accuracy"),
                     ("macro_f1", "Layout macro-F1"), ("reading_order_tau_mean", "Reading-order τ")):
        out.append(f"| {lab} | " + " | ".join(_pct(g["configs"][cfg]["all"].get(key)) for cfg in cfgs) + " |")
    td = lambda cfg: g["configs"][cfg]["all"]["table_detection"]
    out.append("| Table detection F1 | " + " | ".join(_pct(td(cfg)["f1"]) for cfg in cfgs) + " |")
    out.append("")

    out += ["### Full corpus (Auto, against the PDF text layer)", "",
            "| Config | Part | pages | CER median | Numeric fidelity | Pages/min |", "|---|---|---|---|---|---|"]
    for cfg, parts in c["configs"].items():
        for part in ("edgar", "consumer", "simulated_scan"):
            if part not in parts:
                continue
            m = parts[part]
            out.append(f"| {cfg} | {part} | {m['pages']} | {_ci(m.get('cer_median'))} | "
                       f"{_ci(m.get('numeric_fidelity'))} | {m.get('pages_per_minute', 0):.1f} |")
    out.append("")

    out += ["### H6 (scan CER ≥ 2× source CER)", "", "| Config | pairs | scan CER | source CER | ratio | supported |",
            "|---|---|---|---|---|---|"]
    for cfg, h in c["h6"].items():
        out.append(f"| {cfg} | {h['pairs']} | {_pct(h['scan_cer_median'])} | {_pct(h['source_cer_median'])} | "
                   f"{h['ratio']:.2f} | {'yes' if h['supported'] else 'no'} |")
    return "\n".join(out) + "\n"


def render_cleaning(c: dict) -> str:
    """Cleaning + validation tables from run_cleaning.py output."""
    rows = [(cfg, part, a) for cfg, parts in c["configs"].items() for part, a in parts.items()
            if isinstance(a, dict) and a.get("documents")]
    out = ["### Cleaning (production cleaner on Docling output)", "",
           "| Config | Part | docs | pages | chars removed % | over-cleaned pages | number-space fixes | "
           "of which corrupted numbers | table cells | header lines removed |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for cfg, part, a in rows:
        r = a["repairs"]
        hdr = sum(v for k, v in r.items() if k.startswith("header:"))
        oc = a["overclean_rate"]
        out.append(f"| {cfg} | {part} | {a['documents']} | {a['pages']} | {100 * a['chars_removed_rate']:.4f} | "
                   f"{'–' if oc is None else f'{100 * oc:.1f}% of {a['overclean_checked_pages']}'} | "
                   f"{r['table:number_space']} | {r['table:number_corrupted']} | {a['table_cells']} | {hdr} |")
    codes = sorted({k for _, _, a in rows for k in a["docs_with_code"]})
    out += ["", "### Validation (production validator; documents with each issue)", "",
            "| Config | Part | docs | failed (ERROR) | " + " | ".join(codes) + " |",
            "|---|---|---|---|" + "---|" * len(codes)]
    for cfg, part, a in rows:
        out.append(f"| {cfg} | {part} | {a['documents']} | {a['validation_failed_docs']} | "
                   + " | ".join(str(a["docs_with_code"].get(k, 0)) for k in codes) + " |")
    out += ["", "Stage latency per document (ms, p50 / p95): "
            + "; ".join(f"{cfg} {part}: clean {a['latency_ms']['clean_p50']:.2f} / {a['latency_ms']['clean_p95']:.2f}, "
                        f"validate {a['latency_ms']['validate_p50']:.2f} / {a['latency_ms']['validate_p95']:.2f}"
                        for cfg, part, a in rows)]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    print(render(json.loads(path.read_text())))
    if CLEANING.exists():
        print(render_cleaning(json.loads(CLEANING.read_text())))
