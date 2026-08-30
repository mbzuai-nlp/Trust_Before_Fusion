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
```

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
