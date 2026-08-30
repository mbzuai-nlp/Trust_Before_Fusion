# Code

Run commands from the repository root. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Commands that call a model API read credentials from environment variables such
as `OPENAI_API_KEY`; no credentials are stored in the repository.

## Benchmark preparation and validation

Pool CSVs live under `benchmark/pool/`, while their `polluted_image_path` values
are relative to the repository root. Pass `--image_root .` when validating the
pool or building an image manifest.

```bash
python code/qimg7/validate_package.py \
  --package_dir benchmark/pool --image_root . --strict_local_paths
python code/qimg7/build_image_manifest.py --help
python code/qimg7/build_benchmark.py --help
python code/qimg7/validate_outputs.py --help
python code/qimg7/filter_bad_rows.py --help
python code/qimg7/assert_coverage.py --help
python code/qimg7/make_ablation_benchmarks.py --help
python code/qimg7/summarize_regimes_and_predictions.py --help
```

## Image attack construction (T1-T7)

`code/qimg7/pollute_query_images.py` is the script that produced the shipped
attack images. It reads the clean evidence pools in
`benchmark/query_image_evidence/` and writes a pool CSV plus the attack images.

```bash
python -m pip install -r requirements-construction.txt
export GEMINI_API_KEY=...
python code/qimg7/pollute_query_images.py FAVA     # or --all
python code/qimg7/fill_pollution_gaps.py FAVA      # top up short questions
```

Output goes to `data/query_polluted_{evidence,images}/`, outside `benchmark/`,
so a re-run cannot overwrite the published pool. Regenerated images will not be
byte-identical to the shipped ones — T1/T3 call Gemini and T5/T7 run float GPU
kernels. See [`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md) and
[`docs/IMAGE_POLLUTION_METHODS.md`](../docs/IMAGE_POLLUTION_METHODS.md).

## Generation and routing

```bash
python code/generation/run_openai_baseline.py --help
python code/routing/run_answer_consensus.py --help
python code/routing/run_selfcheck_gate.py --help
python code/routing/run_triage_gate.py --help
python code/routing/run_cascaded_router.py --help
python code/routing/run_source_aware_resolver.py --help
python code/routing/run_soft_gated_resolver.py --help
```

## Evaluation and human validation

```bash
python code/eval/judge_against_clean.py --help
python code/eval/make_results_table.py --help
python code/eval/make_paper_tables.py --help
python code/eval/summarize_human_validation.py
python code/eval/generate_image_attack_annotation_dashboard.py --help
python code/eval/compute_image_attack_iaa.py ANNOTATOR_1.csv ANNOTATOR_2.csv OUTPUT.txt
```

### Rebuilding the aggregate tables without API calls

`evaluation/tables/` is derived from the frozen judgments in
`evaluation/judged/`. Both steps are deterministic and call no model API:

```bash
# 1. per-dataset results from the frozen judgments
for ds in alpacafact:20 biography:20 fava:20 longfact:50; do
  d=${ds%%:*}; n=${ds##*:}
  python code/eval/make_results_table.py \
    --judged_jsonls evaluation/judged/gpt4omini/$d/*.judged.jsonl \
    --output_csv evaluation/results/$d/${d}_qimg7_final_results_gpt4omini_${n}.csv \
    --output_md  evaluation/results/$d/${d}_qimg7_final_results_gpt4omini_${n}.md
done

# 2. macro and per-dataset paper tables
python code/eval/make_paper_tables.py \
  --results_csvs evaluation/results/*/*_final_results_gpt4omini_*.csv
```

Use `--verify` on step 2 to check the recomputation against the released
tables without writing anything. By default step 2 is append-only: a row
already present in a released table is copied through byte-for-byte once its
recomputed values are confirmed to match within 1e-12, and the script aborts
rather than overwrite a published value that does not.

## Prompt field boundary

Benchmark rows carry the construction record next to the evidence: `regime`,
`text_status`, `image_status`, `image_pollution_type`, and the nested
`image_evidence.metadata` block (`manipulation_method`, `manipulation_prompt`,
`donor_question_idx`, `rationale`, `original_alt`, `polluted_alt`). None of it
may reach an answer, router, or judge prompt.

`code/common/prompt_fields.py` holds the allow-list and the extraction. Each
component declares which allow-list it uses and keeps its own rendering:

| Component | Script | Allow-list |
|---|---|---|
| answer | `code/generation/run_openai_baseline.py` | `answer` |
| self-check gate | `code/routing/run_selfcheck_gate.py` | `selfcheck_gate` |
| triage gate | `code/routing/run_triage_gate.py` | `triage_gate` |
| source-aware resolver | `code/routing/run_source_aware_resolver.py` | `source_aware` |
| judge | `code/eval/judge_against_clean.py` | `judge` |

All five currently permit the same fields (`question`; `text_evidence.title`,
`.snippet`, `.url`; `image_evidence.title`, `.alt_text`, `.page_url`). The judge
differs by being handed the clean `TC_IC` row for the qid, not by seeing extra
fields. `local_path` / `resolved_local_path` are excluded on purpose: a packaged
path such as `images/FAVA/T6_q18_i1.jpg` names the attack family in its
filename. They are used to load pixels, never rendered into text.

```bash
python code/tests/test_prompt_leakage.py
```

Runs every released row through every builder and fails if a construction field
appears by key or by value. No API key, no network.

## Image manifest and count reconciliation

```bash
python code/qimg7/build_image_hash_manifest.py --counts_only
python code/qimg7/build_image_hash_manifest.py \
  --out metadata/qimg7_image_manifest.csv
```

Reconciles `benchmark/pool/*.csv` against the `images/` tree and emits a
per-image SHA-256 manifest. Hashes are read from the Git LFS pointer when a
file has not been materialised, so this works without `git lfs pull`. The
manifest's rights columns are intentionally empty and require human review.

Each command accepts explicit input and output paths. Use `--help` for its full
argument list.
