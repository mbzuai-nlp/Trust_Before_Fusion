#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -n "$PYTHON_BIN" ]; then
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Provided PYTHON_BIN is not executable in PATH: $PYTHON_BIN"
    exit 1
  fi
elif [ -x "./.venv_mm/bin/python" ]; then
  PYTHON_BIN="./.venv_mm/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "No usable Python interpreter found. Set PYTHON_BIN or create .venv_mm/bin/python."
  exit 1
fi

MODEL_TAG="${MODEL_TAG:-gpt4omini}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
LOGDIR="logs/qimg7_answer_consensus"

# Released layout. Inputs are read from the frozen artifacts; every output is
# written under OUT_ROOT so a re-run cannot overwrite evaluation/predictions/,
# evaluation/judged/, or evaluation/choices/.
EVALUATED_DIR="${QIMG7_EVALUATED_DIR:-benchmark/evaluated}"
PRED_IN_ROOT="${QIMG7_PRED_IN_ROOT:-evaluation/predictions/$MODEL_TAG}"
JUDGED_IN_ROOT="${QIMG7_JUDGED_IN_ROOT:-evaluation/judged/$MODEL_TAG}"
OUT_ROOT="${QIMG7_OUT_ROOT:-data/mm_qimg7_answer_consensus/$MODEL_TAG}"

# Not in the public release; see REP-01 and the open questions in the PR.
SUMMARY_SCRIPT="${QIMG7_DEFENSE_SUMMARY_SCRIPT:-code/eval/summarize_defense_baselines.py}"
SUMMARY_CSV="${QIMG7_DEFENSE_SUMMARY_CSV:-$OUT_ROOT/qimg7_defense_baselines_${MODEL_TAG}.csv}"

mkdir -p "$LOGDIR" "$OUT_ROOT"

# --------------------------------------------------------------------------
# Preflight: fail fast, before any paid API call, naming every absent input.
# --------------------------------------------------------------------------
PREFLIGHT_ERRORS=()
need_file() {
  [ -e "$1" ] || PREFLIGHT_ERRORS+=("missing $2: $1")
}

for s in code/routing/run_answer_consensus.py code/qimg7/validate_outputs.py \
         code/eval/judge_against_clean.py code/eval/make_results_table.py; do
  need_file "$s" "script (run from the repository root)"
done

for spec in LongFact:longfact:50 Biography:biography:20 AlpacaFact:alpacafact:20 FAVA:fava:20; do
  prefix="$(echo "$spec" | cut -d: -f2)"; n="$(echo "$spec" | cut -d: -f3)"
  need_file "$EVALUATED_DIR/${prefix}_qimg7_mm_regimes_${n}.jsonl" "evaluated benchmark"
  for base in parametric text_only full_mm; do
    need_file "$PRED_IN_ROOT/$prefix/${base}_${n}.jsonl" "input prediction"
  done
done

if [ "${#PREFLIGHT_ERRORS[@]}" -gt 0 ]; then
  echo "Preflight failed. Not starting; no API calls were made:" >&2
  for e in "${PREFLIGHT_ERRORS[@]}"; do echo "  - $e" >&2; done
  exit 2
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

echo "NOTE: this script calls the judge API and costs money."
echo "      The released answer_consensus judgments are already in"
echo "      $JUDGED_IN_ROOT/<dataset>/; to rebuild the tables from those"
echo "      without any API call, see code/README.md."

run_dataset() {
  local dataset="$1"
  local prefix="$2"
  local n="$3"
  local outdir="$OUT_ROOT/${prefix}"
  local bench="$EVALUATED_DIR/${prefix}_qimg7_mm_regimes_${n}.jsonl"
  local pred_in="$PRED_IN_ROOT/${prefix}"
  local judged_in="$JUDGED_IN_ROOT/${prefix}"
  local pred_dir="$outdir/predictions"
  local judged_dir="$outdir/judged"
  local answer_pred="$pred_dir/answer_consensus_${n}.jsonl"
  local answer_choices="$pred_dir/answer_consensus_${n}.choices.jsonl"
  local answer_judged="$judged_dir/answer_consensus_${n}.judged.jsonl"
  local result_csv="$outdir/results/${prefix}_qimg7_defense_baselines_${MODEL_TAG}_${n}.csv"
  local result_md="$outdir/results/${prefix}_qimg7_defense_baselines_${MODEL_TAG}_${n}.md"

  mkdir -p "$pred_dir" "$judged_dir" "$outdir/results"
  echo
  echo "========== ANSWER CONSENSUS: $dataset =========="

  "$PYTHON_BIN" code/routing/run_answer_consensus.py \
    --benchmark_jsonl "$bench" \
    --param_preds "$pred_in/parametric_${n}.jsonl" \
    --text_preds "$pred_in/text_only_${n}.jsonl" \
    --full_preds "$pred_in/full_mm_${n}.jsonl" \
    --output_jsonl "$answer_pred" \
    --choices_jsonl "$answer_choices" \
    --baseline_name answer_consensus \
    2>&1 | tee "$LOGDIR/${prefix}_${n}_answer_consensus.log"

  "$PYTHON_BIN" code/qimg7/validate_outputs.py \
    --benchmark_jsonl "$bench" \
    --file "$answer_pred" \
    --kind prediction \
    --strict

  if "$PYTHON_BIN" code/qimg7/validate_outputs.py \
    --benchmark_jsonl "$bench" \
    --file "$answer_judged" \
    --kind judged \
    --strict >/dev/null 2>&1; then
    echo "Judged answer_consensus already valid for $dataset; skipping judge."
  else
    "$PYTHON_BIN" code/eval/judge_against_clean.py \
      --benchmark_jsonl "$bench" \
      --predictions_jsonl "$answer_pred" \
      --output_jsonl "$answer_judged" \
      --model "$JUDGE_MODEL" \
      --api_key_env OPENAI_API_KEY \
      --backend responses \
      --baseline_name answer_consensus \
      --resume \
      --retry_error_rows \
      --request_timeout_s 90 \
      --max_retries 3 \
      --progress_every 50 \
      2>&1 | tee "$LOGDIR/${prefix}_${n}_answer_consensus_judge.log"

    "$PYTHON_BIN" code/qimg7/validate_outputs.py \
      --benchmark_jsonl "$bench" \
      --file "$answer_judged" \
      --kind judged \
      --strict
  fi

  "$PYTHON_BIN" code/eval/make_results_table.py \
    --judged_jsonls \
      "$judged_in/full_mm_${n}.judged.jsonl" \
      "$judged_in/cascaded_router_${n}.judged.jsonl" \
      "$answer_judged" \
      "$judged_in/field_selector_${n}.judged.jsonl" \
    --output_csv "$result_csv" \
    --output_md "$result_md"
}

run_dataset LongFact longfact 50
run_dataset Biography biography 20
run_dataset AlpacaFact alpacafact 20
run_dataset FAVA fava 20

if [ -f "$SUMMARY_SCRIPT" ]; then
  "$PYTHON_BIN" "$SUMMARY_SCRIPT" \
    --model_tag "$MODEL_TAG" \
    --output_csv "$SUMMARY_CSV"
  echo "CSV: $SUMMARY_CSV"
else
  echo
  echo "Skipping the cross-dataset defense-baseline summary: $SUMMARY_SCRIPT" >&2
  echo "is not part of the public release (REP-01). Per-dataset tables were" >&2
  echo "still written under $OUT_ROOT/<dataset>/results/. Set" >&2
  echo "QIMG7_DEFENSE_SUMMARY_SCRIPT once the original is recovered." >&2
fi

echo "DONE answer-consensus baseline"
