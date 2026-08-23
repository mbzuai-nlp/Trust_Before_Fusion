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

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

MODEL_TAG="${MODEL_TAG:-gpt4omini}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
LOGDIR="logs/qimg7_answer_consensus"
mkdir -p "$LOGDIR" "paper_assets/qimg7_final"

run_dataset() {
  local dataset="$1"
  local prefix="$2"
  local n="$3"
  local outdir="data/mm_qimg7_${prefix}"
  local bench="$outdir/${prefix}_qimg7_mm_regimes_${n}.jsonl"
  local pred_dir="$outdir/predictions/$MODEL_TAG"
  local judged_dir="$outdir/judged/$MODEL_TAG"
  local answer_pred="$pred_dir/answer_consensus_${n}.jsonl"
  local answer_choices="$pred_dir/answer_consensus_${n}.choices.jsonl"
  local answer_judged="$judged_dir/answer_consensus_${n}.judged.jsonl"
  local result_csv="$outdir/results/${prefix}_qimg7_defense_baselines_${MODEL_TAG}_${n}.csv"
  local result_md="$outdir/results/${prefix}_qimg7_defense_baselines_${MODEL_TAG}_${n}.md"

  mkdir -p "$pred_dir" "$judged_dir" "$outdir/results"
  echo
  echo "========== ANSWER CONSENSUS: $dataset =========="

  "$PYTHON_BIN" mm_gate_tools/16_run_answer_consensus.py \
    --benchmark_jsonl "$bench" \
    --param_preds "$pred_dir/parametric_${n}.jsonl" \
    --text_preds "$pred_dir/text_only_${n}.jsonl" \
    --full_preds "$pred_dir/full_mm_${n}.jsonl" \
    --output_jsonl "$answer_pred" \
    --choices_jsonl "$answer_choices" \
    --baseline_name answer_consensus \
    2>&1 | tee "$LOGDIR/${prefix}_${n}_answer_consensus.log"

  "$PYTHON_BIN" mm_qimg7_tools/03_validate_jsonl_outputs.py \
    --benchmark_jsonl "$bench" \
    --file "$answer_pred" \
    --kind prediction \
    --strict

  if "$PYTHON_BIN" mm_qimg7_tools/03_validate_jsonl_outputs.py \
    --benchmark_jsonl "$bench" \
    --file "$answer_judged" \
    --kind judged \
    --strict >/dev/null 2>&1; then
    echo "Judged answer_consensus already valid for $dataset; skipping judge."
  else
    "$PYTHON_BIN" mm_eval_tools/06_judge_against_clean.py \
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

    "$PYTHON_BIN" mm_qimg7_tools/03_validate_jsonl_outputs.py \
      --benchmark_jsonl "$bench" \
      --file "$answer_judged" \
      --kind judged \
      --strict
  fi

  "$PYTHON_BIN" mm_eval_tools/07_make_results_table.py \
    --judged_jsonls \
      "$judged_dir/full_mm_${n}.judged.jsonl" \
      "$judged_dir/cascaded_router_${n}.judged.jsonl" \
      "$judged_dir/answer_consensus_${n}.judged.jsonl" \
      "$judged_dir/field_selector_${n}.judged.jsonl" \
    --output_csv "$result_csv" \
    --output_md "$result_md"
}

run_dataset LongFact longfact 50
run_dataset Biography biography 20
run_dataset AlpacaFact alpacafact 20
run_dataset FAVA fava 20

"$PYTHON_BIN" scripts/summarize_qimg7_defense_baselines.py \
  --model_tag "$MODEL_TAG" \
  --output_csv "paper_assets/qimg7_final/qimg7_defense_baselines_gpt4omini.csv"

echo "DONE answer-consensus baseline"
echo "CSV: paper_assets/qimg7_final/qimg7_defense_baselines_gpt4omini.csv"
