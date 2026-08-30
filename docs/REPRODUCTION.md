# Reproducing QIMG-7 results

Two paths. The verification path re-derives every released table from the
frozen artifacts and makes **no API calls**. The regeneration path re-runs the
models and **costs money and changes published numbers**.

Run everything from the repository root.

---

## 1. Verification path — no API key, no network, no cost

```bash
python -m pip install -r requirements.txt
python code/eval/verify_released_artifacts.py
```

Checks, in order:

1. every released prediction and judgment validates against its benchmark
   (`code/qimg7/validate_outputs.py --strict`);
2. regenerating each per-dataset results CSV from `evaluation/judged/`
   reproduces the committed file;
3. regenerating the aggregate tables reproduces every released row to within
   1e-12;
4. the `images/` tree reconciles against `benchmark/pool/*.csv`.

Also available on their own:

```bash
python code/tests/test_prompt_leakage.py                 # prompt field boundary
python code/qimg7/build_image_hash_manifest.py --counts_only
python code/qimg7/validate_package.py \
  --package_dir benchmark/pool --image_root . --strict_local_paths
```

The image checks read SHA-256s from the Git LFS pointers, so they work without
`git lfs pull`. Everything else reads only text artifacts.

---

## 2. What produces each table

| Output | Command | Input | LLM calls |
|---|---|---|---|
| `evaluation/results/<ds>/*_final_results_gpt4omini_<n>.csv` | `code/eval/make_results_table.py` | `evaluation/judged/gpt4omini/<ds>/*.judged.jsonl` | none |
| `evaluation/tables/table_main_qimg7_macro.csv` | `code/eval/make_paper_tables.py` | the four per-dataset results CSVs | none |
| `evaluation/tables/table_per_dataset_qimg7.csv` | `code/eval/make_paper_tables.py` | same | none |
| `evaluation/human_validation/*` summaries | `code/eval/summarize_human_validation.py`, `code/eval/compute_image_attack_iaa.py` | `evaluation/human_validation/` | none |
| `metadata/qimg7_image_manifest.csv` | `code/qimg7/build_image_hash_manifest.py` | `benchmark/pool/`, `images/` | none |

Exact commands are in [`code/README.md`](../code/README.md).

The script that originally produced the two aggregate tables is not in the
repository. `make_paper_tables.py` reproduces every released value to within
1e-12 but a few cells differ in the last one or two ULP, so it is append-only
by default and refuses to overwrite a published value that does not match.

---

## 3. Regeneration path — costs money

> Re-running generation or judging calls a paid API and will change published
> numbers. Do not run it to "check" a result; use the verification path.

### Per-stage cost driver

The evaluated benchmark is **1,760 rows** (LongFact 800, AlpacaFact 320,
Biography 320, FAVA 320). Counts below are one call per row across all four
datasets, taken from the released artifacts.

| Stage | Script | Calls per full run | Released artifact? |
|---|---|---|---|
| `parametric` | `code/generation/run_openai_baseline.py` | 1,760 | yes |
| `text_only` | same | 1,760 | yes |
| `full_mm` | same | 1,760 | yes |
| `selfcheck_gate` | `code/routing/run_selfcheck_gate.py` | 1,760 | yes |
| `triage_gate` | `code/routing/run_triage_gate.py` | 1,760 | **no** |
| `source_aware_selector` | `code/routing/run_source_aware_resolver.py --mode select_only` | 1,760 | yes |
| `source_aware_conductor` | `code/routing/run_source_aware_resolver.py` | 1,760 | yes |
| `cascaded_router` | `code/routing/run_cascaded_router.py` | 0 — deterministic | yes |
| `field_selector` | `code/routing/run_soft_gated_resolver.py` | 0 — deterministic | yes |
| `soft_conductor` | `code/routing/run_soft_gated_resolver.py --policy soft` | 0 — deterministic | yes |
| `answer_consensus` | `code/routing/run_answer_consensus.py` | 0 — deterministic | yes |
| judging | `code/eval/judge_against_clean.py` | 1,760 per baseline (14,080 for the 8 released) | yes |

A full regeneration of the released set is therefore roughly **12,320
generation/routing calls plus 14,080 judge calls**. Token counts and dollar
cost were not recorded for the published run and are not reproduced here rather
than estimated.

The four deterministic stages need no key at all. `answer_consensus` has been
verified to reproduce exactly: running it over the frozen parametric,
text_only, and full_mm predictions regenerates all 800 LongFact rows
field-for-field identical to the released file.

### Running it

```bash
export OPENAI_API_KEY=...
bash code/pipelines/run_qimg7_pipeline.sh FAVA          # or LongFact/Biography/AlpacaFact
```

By default this enters at `benchmark/evaluated/`, seeds every baseline that has
a released artifact, and only calls the API for stages with no released
artifact. Set `QIMG7_BUILD_FROM_SOURCE=1` to rebuild the benchmark from
`benchmark/pool/` — that path needs the text-pollution CSVs, which are not part
of this release, and will refuse to start without them.

Both pipelines run a preflight check first and abort before any API call if an
input is missing.

---

## 4. Models

The published runs used `gpt-4o-mini` for both generation and judging via the
OpenAI Responses API, with the settings recorded in
`metadata/qimg7_implementation_details_addendum.csv`.

**Immutable snapshot IDs for the published runs were not recorded.** `gpt-4o-mini`
is a moving alias, so a re-run today may not use the same weights. This is a
known gap; see the repository's open questions.

The paper also reports GPT-4.1-mini, Qwen2.5-VL-7B-Instruct, and
Llama-3.2-11B-Vision-Instruct. **No artifacts for those models are in this
release**, and this repository does not contain the configuration used to run
them (checkpoint revision, dtype, quantization, and server settings were not
recorded). If you intend to run the open-weight models yourself, note:

* **Llama-3.2-11B-Vision-Instruct** is gated. You must accept the Llama 3.2
  Community License on Hugging Face and be granted access before the weights
  download. The Llama 3.2 Acceptable Use Policy additionally excludes use of
  the multimodal models by individuals or companies domiciled in the European
  Union, so this model may not be usable for reproduction in the EU.
* **Qwen2.5-VL-7B-Instruct** is Apache-2.0 and not gated.

Both are served through an OpenAI-compatible endpoint; point the pipeline at it
with `GENERATION_BACKEND=chat`, `GEN_API_BASE`, and `GEN_API_KEY_ENV`.

---

## 5. Known gaps

Reproduction is currently partial. What is missing, and why it matters:

| Gap | Effect |
|---|---|
| Text-pollution CSVs (`{DS}_clean.csv`, `{DS}_polluted_aligned.csv`) | The benchmark cannot be rebuilt from source; the released `benchmark/evaluated/` is the earliest public entry point. |
| Ablation-benchmark and regime-summary scripts | The parametric and text_only views cannot be rebuilt; the pipeline seeds those baselines from the released predictions instead. |
| T1-T7 attack construction code | The polluted images cannot be regenerated. `docs/IMAGE_POLLUTION_METHODS.md` specifies the method in prose; `requirements-construction.txt` records the toolchain. |
| Cross-model artifacts | GPT-4.1-mini, Qwen2.5-VL, and Llama-3.2-Vision results cannot be checked against anything in this repository. |
| Bootstrap, atomic-decomposition, and evaluator-sensitivity code, with their seeds and stratification IDs | The reported confidence intervals and sensitivity analyses cannot be re-derived. |
| Model snapshot IDs | An exact re-run of the published numbers is not possible even with a key. |
| No tag, release, or DOI | There is no immutable reference for "the released state". |
