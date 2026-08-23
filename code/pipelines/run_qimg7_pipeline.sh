#!/usr/bin/env bash
set -euo pipefail

DATASET="$1"
MAXQ="${2:-default}"
MODEL="${3:-gpt-4o-mini}"

ATTACKS="caption_flip entity_swap semantic_entity_rewrite figstep_typography adversarial_patch image_blend neural_style_transfer"
PKG="external/pollution_output"
LOGDIR="logs/qimg7"
mkdir -p "$LOGDIR"

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
echo "Using Python interpreter: $PYTHON_BIN"

case "$DATASET" in
  LongFact)
    DS="LongFact"; PREFIX="longfact"; BASEDIR="data/mm_longfact"; DEFAULT_N=50;;
  Biography)
    DS="Biography"; PREFIX="biography"; BASEDIR="data/mm_biography"; DEFAULT_N=20;;
  AlpacaFact)
    DS="AlpacaFact"; PREFIX="alpacafact"; BASEDIR="data/mm_alpacafact"; DEFAULT_N=20;;
  FAVA)
    DS="FAVA"; PREFIX="fava"; BASEDIR="data/mm_fava"; DEFAULT_N=20;;
  *)
    echo "Unknown dataset: $DATASET"; exit 1;;
esac

if [ "$MAXQ" = "default" ]; then
  N="$DEFAULT_N"
else
  N="$MAXQ"
fi

MODEL_TAG_SOURCE="${MODEL_TAG_OVERRIDE:-$MODEL}"
MODEL_TAG="$(echo "$MODEL_TAG_SOURCE" | tr -cd '[:alnum:]')"
if [ -z "$MODEL_TAG" ]; then
  MODEL_TAG="model"
fi

GENERATION_BACKEND="${GENERATION_BACKEND:-responses}"
GEN_API_BASE="${GEN_API_BASE:-}"
GEN_API_KEY_ENV="${GEN_API_KEY_ENV:-OPENAI_API_KEY}"
GATE_API_BASE="${GATE_API_BASE:-$GEN_API_BASE}"
GATE_API_KEY_ENV="${GATE_API_KEY_ENV:-$GEN_API_KEY_ENV}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_BACKEND="${JUDGE_BACKEND:-responses}"
JUDGE_API_BASE="${JUDGE_API_BASE:-}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-OPENAI_API_KEY}"

GEN_CLIENT_ARGS=(--backend "$GENERATION_BACKEND" --api_key_env "$GEN_API_KEY_ENV")
if [ -n "$GEN_API_BASE" ]; then
  GEN_CLIENT_ARGS+=(--api_base "$GEN_API_BASE")
fi
GEN_BASELINE_EXTRA_ARGS=()
if [ "${GEN_NO_REMOTE_IMAGES:-0}" = "1" ]; then
  GEN_BASELINE_EXTRA_ARGS+=(--no_remote_images)
fi
if [ "${GEN_FETCH_REMOTE_IMAGES:-0}" = "1" ]; then
  GEN_BASELINE_EXTRA_ARGS+=(--fetch_remote_images)
fi

GATE_CLIENT_ARGS=(--api_key_env "$GATE_API_KEY_ENV")
if [ -n "$GATE_API_BASE" ]; then
  GATE_CLIENT_ARGS+=(--api_base "$GATE_API_BASE")
fi

JUDGE_CLIENT_ARGS=(--api_key_env "$JUDGE_API_KEY_ENV" --backend "$JUDGE_BACKEND")
if [ -n "$JUDGE_API_BASE" ]; then
  JUDGE_CLIENT_ARGS+=(--api_base "$JUDGE_API_BASE")
fi

OUTDIR="data/mm_qimg7_${PREFIX}"
PRED_DIR="$OUTDIR/predictions/$MODEL_TAG"
JUDGED_DIR="$OUTDIR/judged/$MODEL_TAG"
mkdir -p "$PRED_DIR" "$JUDGED_DIR" "$OUTDIR/results" "$OUTDIR/atomic_v2"

BENCH="$OUTDIR/${PREFIX}_qimg7_mm_regimes_${N}.jsonl"
PARAM_BENCH="$OUTDIR/${PREFIX}_qimg7_parametric_${N}.jsonl"
TEXT_BENCH="$OUTDIR/${PREFIX}_qimg7_text_only_${N}.jsonl"

run_step() {
  local name="$1"
  shift
  echo
  echo "========== STEP: $name =========="
  echo "CMD: $*"
  "$@" 2>&1 | tee "$LOGDIR/${PREFIX}_${N}_${name}.log"
}

validate_pred() {
  local file="$1"
  "$PYTHON_BIN" mm_qimg7_tools/03_validate_jsonl_outputs.py \
    --benchmark_jsonl "$BENCH" \
    --file "$file" \
    --kind prediction \
    --strict
}

validate_judged() {
  local file="$1"
  "$PYTHON_BIN" mm_qimg7_tools/03_validate_jsonl_outputs.py \
    --benchmark_jsonl "$BENCH" \
    --file "$file" \
    --kind judged \
    --strict
}

seed_legacy_judged_if_present() {
  local base="$1"
  local legacy="$OUTDIR/judged/${base}_${N}.judged.jsonl"
  local target="$JUDGED_DIR/${base}_${N}.judged.jsonl"
  if [ ! -f "$target" ] && [ -f "$legacy" ]; then
    cp "$legacy" "$target"
    echo "Seeded model-scoped judged file from legacy path: $legacy -> $target"
  fi
}

require_env_key() {
  local env_name="$1"
  local context="$2"
  if [ -z "${!env_name:-}" ]; then
    echo "$env_name is not set; cannot run $context"
    exit 1
  fi
}

safe_predict() {
  local name="$1"
  local outfile="$2"
  shift 2

  set +e
  validate_pred "$outfile" >/dev/null 2>&1
  local pre_vstatus=$?
  set -e
  if [ "$pre_vstatus" -eq 0 ]; then
    echo "Prediction already valid for $name; skipping generation."
    return 0
  fi

  require_env_key "$GEN_API_KEY_ENV" "prediction step: $name"

  for attempt in 1 2 3; do
    echo "Predict $name attempt $attempt"
    set +e
    "$@" 2>&1 | tee "$LOGDIR/${PREFIX}_${N}_${name}_attempt${attempt}.log"
    status=${PIPESTATUS[0]}
    set -e

    if [ "$status" -ne 0 ]; then
      echo "Command failed with status $status"
      sleep 60
    fi

    set +e
    validate_pred "$outfile"
    vstatus=$?
    set -e

    if [ "$vstatus" -eq 0 ]; then
      echo "Prediction validation passed for $name"
      return 0
    fi

    echo "Prediction validation failed for $name. Filtering bad rows and retrying."
    "$PYTHON_BIN" mm_qimg7_tools/04_filter_bad_jsonl.py --input "$outfile" --kind prediction || true
    sleep 60
  done

  echo "FAILED prediction step: $name"
  exit 1
}

safe_judge() {
  local name="$1"
  local outfile="$2"
  shift 2

  set +e
  validate_judged "$outfile" >/dev/null 2>&1
  local pre_vstatus=$?
  set -e
  if [ "$pre_vstatus" -eq 0 ]; then
    echo "Judged file already valid for $name; skipping judging."
    return 0
  fi

  require_env_key "$JUDGE_API_KEY_ENV" "judge step: $name"

  for attempt in 1 2 3; do
    echo "Judge $name attempt $attempt"
    set +e
    "$@" 2>&1 | tee "$LOGDIR/${PREFIX}_${N}_${name}_judge_attempt${attempt}.log"
    status=${PIPESTATUS[0]}
    set -e

    if [ "$status" -ne 0 ]; then
      echo "Judge command failed with status $status"
      sleep 60
    fi

    set +e
    validate_judged "$outfile"
    vstatus=$?
    set -e

    if [ "$vstatus" -eq 0 ]; then
      echo "Judge validation passed for $name"
      return 0
    fi

    echo "Judge validation failed for $name. Filtering bad rows and retrying."
    "$PYTHON_BIN" mm_qimg7_tools/04_filter_bad_jsonl.py --input "$outfile" --kind judged || true
    sleep 60
  done

  echo "FAILED judge step: $name"
  exit 1
}

echo "Running QIMG7 dataset=$DATASET N=$N model=$MODEL"
echo "Generator backend=$GENERATION_BACKEND gen_api_base=${GEN_API_BASE:-default} judge_model=$JUDGE_MODEL judge_backend=$JUDGE_BACKEND"

run_step validate_package \
  "$PYTHON_BIN" mm_qimg7_tools/00_validate_qimg7_package.py \
    --package_dir "$PKG" \
    --strict_local_paths

run_step make_image_jsonl \
  "$PYTHON_BIN" mm_qimg7_tools/01_make_qimg7_image_jsonl_from_csv.py \
    --package_dir "$PKG" \
    --dataset "$DS" \
    --output_jsonl "$OUTDIR/raw_${PREFIX}_qimg7.jsonl" \
    --attacks $ATTACKS \
    --max_clean_per_question 5 \
    --max_polluted_per_attack 3

run_step build_benchmark \
  "$PYTHON_BIN" mm_qimg7_tools/02_build_qimg7_benchmark.py \
    --image_jsonl "$OUTDIR/raw_${PREFIX}_qimg7.jsonl" \
    --clean_text_csv "$BASEDIR/${DS}_clean.csv" \
    --polluted_text_csv "$BASEDIR/${DS}_polluted_aligned.csv" \
    --output_jsonl "$BENCH" \
    --dataset "$DS" \
    --top_k_text 5 \
    --max_questions "$N" \
    --image_pollution_types $ATTACKS

run_step summarize_benchmark \
  "$PYTHON_BIN" mm_wrapup_tools/04_summarize_regimes_and_predictions.py \
    --benchmark_jsonl "$BENCH"

run_step make_parametric \
  "$PYTHON_BIN" mm_wrapup_tools/05_make_ablation_benchmarks.py \
    --input_jsonl "$BENCH" \
    --output_jsonl "$PARAM_BENCH" \
    --mode parametric

run_step make_text_only \
  "$PYTHON_BIN" mm_wrapup_tools/05_make_ablation_benchmarks.py \
    --input_jsonl "$BENCH" \
    --output_jsonl "$TEXT_BENCH" \
    --mode text_only

safe_predict parametric "$PRED_DIR/parametric_${N}.jsonl" \
  "$PYTHON_BIN" 04_run_openai_mm_baseline.py \
    --benchmark_jsonl "$PARAM_BENCH" \
    --output_jsonl "$PRED_DIR/parametric_${N}.jsonl" \
    --model "$MODEL" \
    "${GEN_CLIENT_ARGS[@]}" \
    ${GEN_BASELINE_EXTRA_ARGS[@]+"${GEN_BASELINE_EXTRA_ARGS[@]}"} \
    --sleep_s 0.5 \
    --resume

safe_predict text_only "$PRED_DIR/text_only_${N}.jsonl" \
  "$PYTHON_BIN" 04_run_openai_mm_baseline.py \
    --benchmark_jsonl "$TEXT_BENCH" \
    --output_jsonl "$PRED_DIR/text_only_${N}.jsonl" \
    --model "$MODEL" \
    "${GEN_CLIENT_ARGS[@]}" \
    ${GEN_BASELINE_EXTRA_ARGS[@]+"${GEN_BASELINE_EXTRA_ARGS[@]}"} \
    --sleep_s 0.5 \
    --resume

safe_predict full_mm "$PRED_DIR/full_mm_${N}.jsonl" \
  "$PYTHON_BIN" 04_run_openai_mm_baseline.py \
    --benchmark_jsonl "$BENCH" \
    --output_jsonl "$PRED_DIR/full_mm_${N}.jsonl" \
    --model "$MODEL" \
    "${GEN_CLIENT_ARGS[@]}" \
    ${GEN_BASELINE_EXTRA_ARGS[@]+"${GEN_BASELINE_EXTRA_ARGS[@]}"} \
    --sleep_s 0.5 \
    --resume

for BASE in parametric text_only full_mm; do
  seed_legacy_judged_if_present "$BASE"
  safe_judge "$BASE" "$JUDGED_DIR/${BASE}_${N}.judged.jsonl" \
    "$PYTHON_BIN" mm_eval_tools/06_judge_against_clean.py \
      --benchmark_jsonl "$BENCH" \
      --predictions_jsonl "$PRED_DIR/${BASE}_${N}.jsonl" \
      --output_jsonl "$JUDGED_DIR/${BASE}_${N}.judged.jsonl" \
      --model "$JUDGE_MODEL" \
      "${JUDGE_CLIENT_ARGS[@]}" \
      --baseline_name "$BASE" \
      --resume \
      --retry_error_rows \
      --request_timeout_s 90 \
      --max_retries 3 \
      --progress_every 50
done

safe_predict selfcheck_gate "$PRED_DIR/selfcheck_gate_${N}.jsonl" \
  "$PYTHON_BIN" mm_gate_tools/02_run_selfcheck_gate.py \
    --benchmark_jsonl "$BENCH" \
    --full_judged "$JUDGED_DIR/full_mm_${N}.judged.jsonl" \
    --param_judged "$JUDGED_DIR/parametric_${N}.judged.jsonl" \
    --output_jsonl "$PRED_DIR/selfcheck_gate_${N}.jsonl" \
    --choices_jsonl "$PRED_DIR/selfcheck_gate_${N}.choices.jsonl" \
    --model "$MODEL" \
    "${GATE_CLIENT_ARGS[@]}" \
    --resume

seed_legacy_judged_if_present selfcheck_gate
safe_judge selfcheck_gate "$JUDGED_DIR/selfcheck_gate_${N}.judged.jsonl" \
  "$PYTHON_BIN" mm_eval_tools/06_judge_against_clean.py \
    --benchmark_jsonl "$BENCH" \
    --predictions_jsonl "$PRED_DIR/selfcheck_gate_${N}.jsonl" \
    --output_jsonl "$JUDGED_DIR/selfcheck_gate_${N}.judged.jsonl" \
    --model "$JUDGE_MODEL" \
    "${JUDGE_CLIENT_ARGS[@]}" \
    --baseline_name selfcheck_gate \
    --resume \
    --retry_error_rows \
    --request_timeout_s 90 \
    --max_retries 3 \
    --progress_every 50

safe_predict triage_gate "$PRED_DIR/triage_gate_${N}.jsonl" \
  "$PYTHON_BIN" mm_gate_tools/06_run_triage_gate.py \
    --benchmark_jsonl "$BENCH" \
    --full_preds "$PRED_DIR/full_mm_${N}.jsonl" \
    --text_preds "$PRED_DIR/text_only_${N}.jsonl" \
    --param_preds "$PRED_DIR/parametric_${N}.jsonl" \
    --output_jsonl "$PRED_DIR/triage_gate_${N}.jsonl" \
    --choices_jsonl "$PRED_DIR/triage_gate_${N}.choices.jsonl" \
    --model "$MODEL" \
    "${GATE_CLIENT_ARGS[@]}" \
    --resume

seed_legacy_judged_if_present triage_gate
safe_judge triage_gate "$JUDGED_DIR/triage_gate_${N}.judged.jsonl" \
  "$PYTHON_BIN" mm_eval_tools/06_judge_against_clean.py \
    --benchmark_jsonl "$BENCH" \
    --predictions_jsonl "$PRED_DIR/triage_gate_${N}.jsonl" \
    --output_jsonl "$JUDGED_DIR/triage_gate_${N}.judged.jsonl" \
    --model "$JUDGE_MODEL" \
    "${JUDGE_CLIENT_ARGS[@]}" \
    --baseline_name triage_gate \
    --resume \
    --retry_error_rows \
    --request_timeout_s 90 \
    --max_retries 3 \
    --progress_every 50

safe_predict cascaded_router "$PRED_DIR/cascaded_router_${N}.jsonl" \
  "$PYTHON_BIN" mm_gate_tools/08_run_cascaded_router.py \
    --benchmark_jsonl "$BENCH" \
    --selfcheck_choices "$PRED_DIR/selfcheck_gate_${N}.choices.jsonl" \
    --triage_choices "$PRED_DIR/triage_gate_${N}.choices.jsonl" \
    --full_preds "$PRED_DIR/full_mm_${N}.jsonl" \
    --text_preds "$PRED_DIR/text_only_${N}.jsonl" \
    --param_preds "$PRED_DIR/parametric_${N}.jsonl" \
    --output_jsonl "$PRED_DIR/cascaded_router_${N}.jsonl" \
    --choices_jsonl "$PRED_DIR/cascaded_router_${N}.choices.jsonl"

seed_legacy_judged_if_present cascaded_router
safe_judge cascaded_router "$JUDGED_DIR/cascaded_router_${N}.judged.jsonl" \
  "$PYTHON_BIN" mm_eval_tools/06_judge_against_clean.py \
    --benchmark_jsonl "$BENCH" \
    --predictions_jsonl "$PRED_DIR/cascaded_router_${N}.jsonl" \
    --output_jsonl "$JUDGED_DIR/cascaded_router_${N}.judged.jsonl" \
    --model "$JUDGE_MODEL" \
    "${JUDGE_CLIENT_ARGS[@]}" \
    --baseline_name cascaded_router \
    --resume \
    --retry_error_rows \
    --request_timeout_s 90 \
    --max_retries 3 \
    --progress_every 50

safe_predict source_aware_selector "$PRED_DIR/source_aware_selector_${N}.jsonl" \
  "$PYTHON_BIN" mm_gate_tools/12_run_source_aware_conductor.py \
    --benchmark_jsonl "$BENCH" \
    --param_preds "$PRED_DIR/parametric_${N}.jsonl" \
    --text_preds "$PRED_DIR/text_only_${N}.jsonl" \
    --full_preds "$PRED_DIR/full_mm_${N}.jsonl" \
    --output_jsonl "$PRED_DIR/source_aware_selector_${N}.jsonl" \
    --choices_jsonl "$PRED_DIR/source_aware_selector_${N}.choices.jsonl" \
    --model "$MODEL" \
    "${GATE_CLIENT_ARGS[@]}" \
    --mode select_only \
    --resume

safe_predict source_aware_conductor "$PRED_DIR/source_aware_conductor_${N}.jsonl" \
  "$PYTHON_BIN" mm_gate_tools/12_run_source_aware_conductor.py \
    --benchmark_jsonl "$BENCH" \
    --param_preds "$PRED_DIR/parametric_${N}.jsonl" \
    --text_preds "$PRED_DIR/text_only_${N}.jsonl" \
    --full_preds "$PRED_DIR/full_mm_${N}.jsonl" \
    --output_jsonl "$PRED_DIR/source_aware_conductor_${N}.jsonl" \
    --choices_jsonl "$PRED_DIR/source_aware_conductor_${N}.choices.jsonl" \
    --model "$MODEL" \
    "${GATE_CLIENT_ARGS[@]}" \
    --mode compose \
    --resume

safe_predict field_selector "$PRED_DIR/field_selector_${N}.jsonl" \
  "$PYTHON_BIN" mm_gate_tools/14_run_soft_gated_resolver.py \
    --benchmark_jsonl "$BENCH" \
    --selfcheck_choices "$PRED_DIR/selfcheck_gate_${N}.choices.jsonl" \
    --param_preds "$PRED_DIR/parametric_${N}.jsonl" \
    --resolver_preds "$PRED_DIR/source_aware_selector_${N}.jsonl" \
    --resolver_choices "$PRED_DIR/source_aware_selector_${N}.choices.jsonl" \
    --output_jsonl "$PRED_DIR/field_selector_${N}.jsonl" \
    --choices_jsonl "$PRED_DIR/field_selector_${N}.choices.jsonl" \
    --baseline_name field_selector \
    --policy field

safe_predict soft_conductor "$PRED_DIR/soft_conductor_${N}.jsonl" \
  "$PYTHON_BIN" mm_gate_tools/14_run_soft_gated_resolver.py \
    --benchmark_jsonl "$BENCH" \
    --selfcheck_choices "$PRED_DIR/selfcheck_gate_${N}.choices.jsonl" \
    --param_preds "$PRED_DIR/parametric_${N}.jsonl" \
    --resolver_preds "$PRED_DIR/source_aware_conductor_${N}.jsonl" \
    --resolver_choices "$PRED_DIR/source_aware_conductor_${N}.choices.jsonl" \
    --output_jsonl "$PRED_DIR/soft_conductor_${N}.jsonl" \
    --choices_jsonl "$PRED_DIR/soft_conductor_${N}.choices.jsonl" \
    --baseline_name soft_conductor \
    --policy soft

for BASE in field_selector soft_conductor; do
  seed_legacy_judged_if_present "$BASE"
  safe_judge "$BASE" "$JUDGED_DIR/${BASE}_${N}.judged.jsonl" \
    "$PYTHON_BIN" mm_eval_tools/06_judge_against_clean.py \
      --benchmark_jsonl "$BENCH" \
      --predictions_jsonl "$PRED_DIR/${BASE}_${N}.jsonl" \
      --output_jsonl "$JUDGED_DIR/${BASE}_${N}.judged.jsonl" \
      --model "$JUDGE_MODEL" \
      "${JUDGE_CLIENT_ARGS[@]}" \
      --baseline_name "$BASE" \
      --resume \
      --retry_error_rows \
      --request_timeout_s 90 \
      --max_retries 3 \
      --progress_every 50
done

run_step final_results \
      "$PYTHON_BIN" mm_eval_tools/07_make_results_table.py \
    --judged_jsonls \
      "$JUDGED_DIR/parametric_${N}.judged.jsonl" \
      "$JUDGED_DIR/text_only_${N}.judged.jsonl" \
      "$JUDGED_DIR/full_mm_${N}.judged.jsonl" \
      "$JUDGED_DIR/cascaded_router_${N}.judged.jsonl" \
      "$JUDGED_DIR/field_selector_${N}.judged.jsonl" \
      "$JUDGED_DIR/soft_conductor_${N}.judged.jsonl" \
    --output_csv "$OUTDIR/results/${PREFIX}_qimg7_final_results_${MODEL_TAG}_${N}.csv" \
    --output_md "$OUTDIR/results/${PREFIX}_qimg7_final_results_${MODEL_TAG}_${N}.md"

echo "DONE QIMG7 $DATASET N=$N"
