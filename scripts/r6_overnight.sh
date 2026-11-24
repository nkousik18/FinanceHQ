#!/bin/zsh
# R6 overnight chain (S-2026-11-24-29). Waits for the Groq daily reset (00:00 UTC = 19:00 EST), then:
#   1. dev answers for the 24 calibration questions (C2-hybrid+rerank, C1-dense)   ~79k tokens
#   2. freeze the calibration sample (first 30 non-refused answers are v1)          0 tokens
#   3. judge the 30 sampled answers                                                 ~87k tokens
#   4. continue S4 test generation with what's left of the budget (resumes later)
#   5. regenerate the S4 report (PARTIAL until all 160 test answers exist)
# Every step stops cleanly at the ledger's daily budget; re-running continues where it stopped.
#   caffeinate -i scripts/r6_overnight.sh          # waits for the reset
#   caffeinate -i scripts/r6_overnight.sh --now    # no wait (budget already reset)
set -e
cd "${0:A:h}/.."
PY=venv/bin/python
CFG=evals/configs/s4_generation.yaml
RUNS="C2-hybrid+rerank,C1-dense"
CAL=evals/golden/judge_calibration

if [[ "$1" != "--now" ]]; then
  now=$(date -u +%s)
  target=$(( (now / 86400 + 1) * 86400 + 120 ))      # next 00:00 UTC + 2 min margin
  echo "waiting $(( (target - now) / 60 )) min for the UTC day reset..."
  sleep $(( target - now ))
fi

echo "== 1. dev generation (calibration questions)"
$PY -m evals.generation.run --config $CFG --split dev --runs "$RUNS" --qids-file $CAL/questions.json --confirm-cost

echo "== 2. freeze calibration sample"
[[ -f $CAL/sample.json ]] || $PY -m evals.generation.calibration sample   # refuses if step 1 is unfinished

echo "== 3. judge calibration sample (30)"
$PY -m evals.generation.judge --config $CFG --split dev --sample 30 --confirm-cost

echo "== 4. S4 test generation (resumes)"
$PY -m evals.generation.run --config $CFG --split test --runs "$RUNS" --confirm-cost

echo "== 5. S4 report"
$PY -m evals.generation.report --split test
echo "done. Label: $PY -m evals.generation.calibration_app.server  (http://127.0.0.1:8767)"
