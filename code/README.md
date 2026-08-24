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
python code/eval/summarize_human_validation.py
python code/eval/generate_image_attack_annotation_dashboard.py --help
python code/eval/compute_image_attack_iaa.py ANNOTATOR_1.csv ANNOTATOR_2.csv OUTPUT.txt
```

Each command accepts explicit input and output paths. Use `--help` for its full
argument list.
