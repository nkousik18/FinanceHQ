# S0 Results — Ingestion and Parsing (R1)

**Status:** pre-registration only. Written 2026-11-06, **before** S0 is run on the verified golden set.
Results get added below this section after the run. Numbers come from `data/results/s0/summary.json` and MLflow, never typed by hand (F7).

Scope: `loandoc-retrieval-evals-scope-v2.md` §8.1 (metrics), §9.3 (S0), §9.6 (statistics), §11 (hypotheses).

---

## 1. Pre-registered hypotheses (copied verbatim from scope v2 §11)

- **H1:** C4 (layout-aware) beats C1 on `recall@budget` for `table` questions.
- **H2:** `R-hybrid` beats `R-dense` on `lookup` questions that contain exact identifiers (rates, amounts, section numbers, defined terms).
- **H3:** Reranking improves `mrr@10` for every base retriever.
- **H4:** `R-pc` improves faithfulness on `explain` questions compared with the same chunker under `R-dense`.
- **H5:** The best configuration's $ per 1k queries is within 20% of the baseline's, because the context budget is fixed.
- **H6:** Simulated-scan pages have at least twice the CER of the digital pages they were rendered from.
- **H7:** Test questions from the lowest ingestion-score tercile have lower answer correctness than those from the highest tercile.
- **H8:** C1's embedder truncation rate is above 50%, and C2–C5's are below 5%.
- **H9 (new):** Under the baseline, amendment questions have the lowest `recall@budget` of any question subgroup, and a stale-answer rate above 20%.
- **H10 (new):** P2 (accurate table mode) beats P1 on cell accuracy for the table-heavy stratum.

Each one is reported as **supported**, **not supported** or **inconclusive**. Any hypothesis added after results are seen is labelled *post-hoc*.

### Which ones S0 tests

| H | Tested in | S0 role |
|---|---|---|
| H6 | **S0** | Tested here (see §2) |
| H10 | **S0** | Tested here (see §2) |
| H7 | S4 (needs answer correctness) | S0 only produces the document ingestion scores H7 uses (Tesseract run, per S-2026-11-06-1) |
| H8 | R2 / S1 (needs C2–C5) | C1 part already measured in R0 (`docs/r0_findings.md`) |
| H1–H5, H9 | S1–S4 | Not tested here |

## 2. Test rules for S0 (fixed before the run)

### S0 decision (D3): P1 vs P2
- **Metric:** ★ cell accuracy (non-empty labelled cells; a missed table scores 0).
- **Test:** paired bootstrap over labelled tables (10,000 resamples, 95% CI) of P2 − P1, all 30 verified pages.
- **Rule:** the winner is the config whose side the CI is on. If the CI includes 0, it's a tie and the faster config (pages per minute on the full corpus) wins.

### H10: P2 beats P1 on cell accuracy, table-heavy stratum
- **Same test as the decision, restricted to tables on the 8 `table_heavy` pages** (about 9 labelled tables).
- **Supported:** CI for P2 − P1 is entirely above 0. **Not supported:** CI entirely below 0. **Inconclusive:** CI includes 0, reported as "no detectable difference at n = …".
- Expect low power: about 9 tables.

### H6: scan CER ≥ 2× source-page CER
- **Pairs:** each simulated-scan page vs. the digital page it was rendered from, in the same parser config. CER is against the source page's PDF text layer (Auto, full corpus; no golden labels involved).
- **Configs:** P1 and P2 (Tesseract, the OCR engine fixed in D10). Apple Vision (P1V/P2V) is reported descriptively, not as the H6 test.
- **Rule:** supported in a config if median scan CER ≥ 2 × median source CER. Reported by scan level (light / medium / heavy), with order-free word recall next to CER.

> **Disclosure: H6 isn't a clean pre-registration.** H6 uses only Auto data, so its outcome was already visible in the R1 dry runs. The pairing rule (each scan vs. its own source page) was also set on 2026-11-04 *after* a first version gave a misleading result (Decision Log, 2026-11-04). The dry run showed H6 **not supported**, because digital-page CER is inflated by text-layer reading order. The H6 result will be reported with this disclosure attached.

H10 and the P1-vs-P2 decision have **not** been seen on verified labels. Dry runs used pre-filled labels, which are P1's own output, so they say nothing about P1 vs P2.

## 3. Reporting commitments
- ★ metrics (CER median, numeric fidelity, cell accuracy, section-header recall) with bootstrap CIs over pages; everything else is labelled diagnostic.
- **n pages per stratum** next to every per-stratum number; per-stratum results are descriptive.
- The **5 from-scratch pages and 25 pre-filled pages are reported separately**, because pre-filled labels may be biased toward Docling P1.
- Split by corpus part (EDGAR / CFPB / simulated scan) where it applies.
- Known gaps stated: split-table rate (needs cross-page labels); cleaning metrics, per-stage latency beyond parsing, and validation counts (need the Docling → `CleanedResult` adapter).

---

## 4. Results

*(Filled in after the S0 run.)*
