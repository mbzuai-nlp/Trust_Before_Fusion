#!/usr/bin/env bash
set -euo pipefail

DATASET="$1"
MAXQ="${2:-default}"
MODEL="${3:-gpt-4o-mini}"

ATTACKS="caption_flip entity_swap semantic_entity_rewrite figstep_typography adversarial_patch image_blend neural_style_transfer"

# Released layout. QIMG7_PKG/QIMG7_IMAGE_ROOT exist so an internal checkout with
# a different arrangement can still drive this script.
PKG="${QIMG7_PKG:-benchmark/pool}"
IMAGE_ROOT="${QIMG7_IMAGE_ROOT:-.}"
EVALUATED_DIR="${QIMG7_EVALUATED_DIR:-benchmark/evaluated}"

# Frozen artifacts from the released run. These are read-only inputs: the
# pipeline copies from them into its own output tree and never writes back.
RELEASED_PRED_ROOT="${QIMG7_RELEASED_PRED_ROOT:-evaluation/predictions}"
RELEASED_JUDGED_ROOT="${QIMG7_RELEASED_JUDGED_ROOT:-evaluation/judged}"

# Default entry point is the released benchmark under benchmark/evaluated/.
# The construction stages (package validation -> image manifest -> benchmark
# build -> ablation benchmarks) need the text-pollution CSVs, which are not part
# of the public release, so they are opt-in.
BUILD_FROM_SOURCE="${QIMG7_BUILD_FROM_SOURCE:-0}"

# Not in the public release; see REP-01 and the open questions in the PR. Point
# these at the originals once they are recovered and the stages will run.
SUMMARIZE_SCRIPT="${QIMG7_SUMMARIZE_SCRIPT:-code/qimg7/summarize_regimes_and_predictions.py}"
ABLATION_SCRIPT="${QIMG7_ABLATION_SCRIPT:-code/qimg7/make_ablation_benchmarks.py}"

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

OUTDIR="${QIMG7_OUTDIR:-data/mm_qimg7_${PREFIX}}"
PRED_DIR="$OUTDIR/predictions/$MODEL_TAG"
JUDGED_DIR="$OUTDIR/judged/$MODEL_TAG"
mkdir -p "$PRED_DIR" "$JUDGED_DIR" "$OUTDIR/results" "$OUTDIR/atomic_v2"

if [ "$BUILD_FROM_SOURCE" = "1" ]; then
  BENCH="$OUTDIR/${PREFIX}_qimg7_mm_regimes_${N}.jsonl"
else
  BENCH="$EVALUATED_DIR/${PREFIX}_qimg7_mm_regimes_${N}.jsonl"
fi
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
  "$PYTHON_BIN" code/qimg7/validate_outputs.py \
    --benchmark_jsonl "$BENCH" \
    --file "$file" \
    --kind prediction \
    --strict
}

validate_judged() {
  local file="$1"
  "$PYTHON_BIN" code/qimg7/validate_outputs.py \
    --benchmark_jsonl "$BENCH" \
    --file "$file" \
    --kind judged \
    --strict
}

# Copy a frozen released artifact into this run's output tree if we do not have
# one yet. Never writes into evaluation/.
seed_from_released() {
  local kind="$1"    # predictions | judged
  local base="$2"
  local src dest
  if [ "$kind" = "predictions" ]; then
    src="$RELEASED_PRED_ROOT/$MODEL_TAG/$PREFIX/${base}_${N}.jsonl"
    dest="$PRED_DIR/${base}_${N}.jsonl"
  else
    src="$RELEASED_JUDGED_ROOT/$MODEL_TAG/$PREFIX/${base}_${N}.judged.jsonl"
    dest="$JUDGED_DIR/${base}_${N}.judged.jsonl"
  fi
  if [ ! -f "$dest" ] && [ -f "$src" ]; then
    cp "$src" "$dest"
    echo "Seeded $kind for $base from the released artifacts: $src"
    if [ "$kind" = "predictions" ]; then
      local choices="${src%.jsonl}.choices.jsonl"
      if [ -f "$choices" ]; then
        cp "$choices" "${dest%.jsonl}.choices.jsonl"
      fi
    fi
  fi
  return 0
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
    "$PYTHON_BIN" code/qimg7/filter_bad_rows.py --input "$outfile" --kind prediction || true
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
    "$PYTHON_BIN" code/qimg7/filter_bad_rows.py --input "$outfile" --kind judged || true
    sleep 60
  done

  echo "FAILED judge step: $name"
  exit 1
}

# ---------------------------------------------------------------------------
# Preflight: fail fast, before any API call, naming every absent prerequisite.
# ---------------------------------------------------------------------------
PREFLIGHT_ERRORS=()

need_file() {
  [ -e "$1" ] || PREFLIGHT_ERRORS+=("missing $2: $1${3:+ -- $3}")
}

for s in code/qimg7/validate_outputs.py code/qimg7/filter_bad_rows.py \
         code/generation/run_openai_baseline.py code/eval/judge_against_clean.py \
         code/eval/make_results_table.py code/routing/run_selfcheck_gate.py \
         code/routing/run_triage_gate.py code/routing/run_cascaded_router.py \
         code/routing/run_source_aware_resolver.py \
         code/routing/run_soft_gated_resolver.py; do
  need_file "$s" "script" "run this from the repository root"
done

if [ "$BUILD_FROM_SOURCE" = "1" ]; then
  need_file "$PKG" "pool directory (QIMG7_PKG)"
  need_file "$BASEDIR/${DS}_clean.csv" "clean text CSV" \
    "the text-pollution CSVs are not part of the public release"
  need_file "$BASEDIR/${DS}_polluted_aligned.csv" "polluted text CSV" \
    "the text-pollution CSVs are not part of the public release"
  need_file "$SUMMARIZE_SCRIPT" "summarize script" \
    "not released; set QIMG7_SUMMARIZE_SCRIPT once recovered (REP-01)"
  need_file "$ABLATION_SCRIPT" "ablation-benchmark script" \
    "not released; set QIMG7_ABLATION_SCRIPT once recovered (REP-01)"
else
  need_file "$BENCH" "evaluated benchmark" \
    "expected the released benchmark for $DATASET at N=$N"
  # parametric and text_only run on ablation views of the benchmark. Those
  # views need the unreleased ablation script -- but the released run's own
  # predictions for both baselines are in the repo, so seed from those instead
  # and only demand the script when neither route is available.
  for base in parametric text_only; do
    bench_var="$([ "$base" = parametric ] && echo "$PARAM_BENCH" || echo "$TEXT_BENCH")"
    if [ ! -f "$bench_var" ] \
       && [ ! -f "$RELEASED_PRED_ROOT/$MODEL_TAG/$PREFIX/${base}_${N}.jsonl" ]; then
      need_file "$ABLATION_SCRIPT" "ablation-benchmark script" \
        "needed to build $bench_var for the $base baseline, and no released prediction to seed from; not released (REP-01)"
    fi
  done
fi

if [ "${#PREFLIGHT_ERRORS[@]}" -gt 0 ]; then
  echo "Preflight failed. This pipeline cannot run as configured:" >&2
  for e in "${PREFLIGHT_ERRORS[@]}"; do echo "  - $e" >&2; done
  echo >&2
  echo "The public release ships the evaluated benchmark (benchmark/evaluated/)," >&2
  echo "the frozen predictions and judgments (evaluation/), and the pool CSVs" >&2
  echo "(benchmark/pool/). It does not ship the text-pollution CSVs or the" >&2
  echo "ablation/summary scripts, so the construction stages cannot run from a" >&2
  echo "public checkout. To rebuild the aggregate tables from the frozen" >&2
  echo "judgments with no API calls, see 'Rebuilding the aggregate tables'" >&2
  echo "in code/README.md." >&2
  exit 2
fi

echo "Running QIMG7 dataset=$DATASET N=$N model=$MODEL"
echo "Entry point: $([ "$BUILD_FROM_SOURCE" = "1" ] && echo 'build from source' || echo 'released benchmark')"
echo "Benchmark: $BENCH"
echo "Generator backend=$GENERATION_BACKEND gen_api_base=${GEN_API_BASE:-default} judge_model=$JUDGE_MODEL judge_backend=$JUDGE_BACKEND"

if [ "$BUILD_FROM_SOURCE" = "1" ]; then
  # Pool paths (polluted_image_path) are relative to the repository root, so the
  # package dir and the image root are separate arguments under this layout.
  run_step validate_package \
    "$PYTHON_BIN" code/qimg7/validate_package.py \
      --package_dir "$PKG" \
      --image_root "$IMAGE_ROOT" \
      --strict_local_paths

  run_step make_image_jsonl \
    "$PYTHON_BIN" code/qimg7/build_image_manifest.py \
      --package_dir "$PKG" \
      --image_root "$IMAGE_ROOT" \
      --dataset "$DS" \
      --output_jsonl "$OUTDIR/raw_${PREFIX}_qimg7.jsonl" \
      --attacks $ATTACKS \
      --max_clean_per_question 5 \
      --max_polluted_per_attack 3

  run_step build_benchmark \
    "$PYTHON_BIN" code/qimg7/build_benchmark.py \
      --image_jsonl "$OUTDIR/raw_${PREFIX}_qimg7.jsonl" \
      --clean_text_csv "$BASEDIR/${DS}_clean.csv" \
      --polluted_text_csv "$BASEDIR/${DS}_polluted_aligned.csv" \
      --output_jsonl "$BENCH" \
      --dataset "$DS" \
      --top_k_text 5 \
      --max_questions "$N" \
      --image_pollution_types $ATTACKS

  run_step summarize_benchmark \
    "$PYTHON_BIN" "$SUMMARIZE_SCRIPT" \
      --benchmark_jsonl "$BENCH"
else
  echo "Entering at the released benchmark; construction stages skipped."
  echo "Set QIMG7_BUILD_FROM_SOURCE=1 to rebuild from benchmark/pool/ (needs the"
  echo "unreleased text-pollution CSVs)."
fi

# The parametric and text_only baselines read ablation views of the benchmark.
# When we cannot build those views, fall back to the released predictions for
# the same baselines, which validate against this benchmark unchanged.
if [ "$BUILD_FROM_SOURCE" != "1" ]; then
  # Every baseline with a frozen artifact in the release. triage_gate is
  # deliberately absent: this pipeline runs it, but no triage_gate prediction or
  # judgment was published, so that step still needs an API key.
  for base in parametric text_only full_mm selfcheck_gate cascaded_router \
              source_aware_conductor source_aware_selector field_selector \
              soft_conductor answer_consensus; do
    seed_from_released predictions "$base"
    seed_from_released judged "$base"
  done
fi

if [ ! -f "$PARAM_BENCH" ] && [ ! -f "$PRED_DIR/parametric_${N}.jsonl" ]; then
  run_step make_parametric \
    "$PYTHON_BIN" "$ABLATION_SCRIPT" \
      --input_jsonl "$BENCH" \
      --output_jsonl "$PARAM_BENCH" \
      --mode parametric
fi

if [ ! -f "$TEXT_BENCH" ] && [ ! -f "$PRED_DIR/text_only_${N}.jsonl" ]; then
  run_step make_text_only \
    "$PYTHON_BIN" "$ABLATION_SCRIPT" \
      --input_jsonl "$BENCH" \
      --output_jsonl "$TEXT_BENCH" \
      --mode text_only
fi

safe_predict parametric "$PRED_DIR/parametric_${N}.jsonl" \
  "$PYTHON_BIN" code/generation/run_openai_baseline.py \
    --benchmark_jsonl "$PARAM_BENCH" \
    --output_jsonl "$PRED_DIR/parametric_${N}.jsonl" \
    --model "$MODEL" \
    "${GEN_CLIENT_ARGS[@]}" \
    ${GEN_BASELINE_EXTRA_ARGS[@]+"${GEN_BASELINE_EXTRA_ARGS[@]}"} \
    --sleep_s 0.5 \
    --resume

safe_predict text_only "$PRED_DIR/text_only_${N}.jsonl" \
  "$PYTHON_BIN" code/generation/run_openai_baseline.py \
    --benchmark_jsonl "$TEXT_BENCH" \
    --output_jsonl "$PRED_DIR/text_only_${N}.jsonl" \
    --model "$MODEL" \
    "${GEN_CLIENT_ARGS[@]}" \
    ${GEN_BASELINE_EXTRA_ARGS[@]+"${GEN_BASELINE_EXTRA_ARGS[@]}"} \
    --sleep_s 0.5 \
    --resume

safe_predict full_mm "$PRED_DIR/full_mm_${N}.jsonl" \
  "$PYTHON_BIN" code/generation/run_openai_baseline.py \
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
    "$PYTHON_BIN" code/eval/judge_against_clean.py \
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
  "$PYTHON_BIN" code/routing/run_selfcheck_gate.py \
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
  "$PYTHON_BIN" code/eval/judge_against_clean.py \
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

# triage_gate is an intermediate stage: its only consumer is cascaded_router.
# No triage_gate artifact was published, so on the default path cascaded_router
# is seeded from the release and triage_gate has nothing left to feed. Re-running
# it would spend money to produce an input nothing reads.
if [ -f "$PRED_DIR/cascaded_router_${N}.jsonl" ] \
   && [ ! -f "$PRED_DIR/triage_gate_${N}.choices.jsonl" ]; then
  TRIAGE_NEEDED=0
  echo "Skipping triage_gate: cascaded_router is already available, and"
  echo "triage_gate's only consumer is cascaded_router. Delete"
  echo "$PRED_DIR/cascaded_router_${N}.jsonl to force both to re-run."
else
  TRIAGE_NEEDED=1
fi

if [ "$TRIAGE_NEEDED" = "1" ]; then
safe_predict triage_gate "$PRED_DIR/triage_gate_${N}.jsonl" \
  "$PYTHON_BIN" code/routing/run_triage_gate.py \
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
  "$PYTHON_BIN" code/eval/judge_against_clean.py \
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
fi

safe_predict cascaded_router "$PRED_DIR/cascaded_router_${N}.jsonl" \
  "$PYTHON_BIN" code/routing/run_cascaded_router.py \
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
  "$PYTHON_BIN" code/eval/judge_against_clean.py \
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
  "$PYTHON_BIN" code/routing/run_source_aware_resolver.py \
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
  "$PYTHON_BIN" code/routing/run_source_aware_resolver.py \
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
  "$PYTHON_BIN" code/routing/run_soft_gated_resolver.py \
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
  "$PYTHON_BIN" code/routing/run_soft_gated_resolver.py \
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
    "$PYTHON_BIN" code/eval/judge_against_clean.py \
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

# Summarise every baseline that actually has a judged file, so the table matches
# what this run produced. answer_consensus comes from run_answer_consensus.sh and
# is included when its judgments are present.
FINAL_JUDGED=()
for BASE in parametric text_only full_mm selfcheck_gate triage_gate \
            cascaded_router field_selector soft_conductor answer_consensus; do
  if [ -f "$JUDGED_DIR/${BASE}_${N}.judged.jsonl" ]; then
    FINAL_JUDGED+=("$JUDGED_DIR/${BASE}_${N}.judged.jsonl")
  fi
done

if [ "${#FINAL_JUDGED[@]}" -eq 0 ]; then
  echo "No judged files under $JUDGED_DIR; nothing to summarise." >&2
  exit 1
fi

run_step final_results \
      "$PYTHON_BIN" code/eval/make_results_table.py \
    --judged_jsonls "${FINAL_JUDGED[@]}" \
    --output_csv "$OUTDIR/results/${PREFIX}_qimg7_final_results_${MODEL_TAG}_${N}.csv" \
    --output_md "$OUTDIR/results/${PREFIX}_qimg7_final_results_${MODEL_TAG}_${N}.md"

echo "DONE QIMG7 $DATASET N=$N"
