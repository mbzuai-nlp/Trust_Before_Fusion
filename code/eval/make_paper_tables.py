#!/usr/bin/env python3
"""Aggregate per-dataset result CSVs into the paper's macro and per-dataset tables.

Input is one ``*_final_results_*.csv`` per dataset, as produced by
``code/eval/make_results_table.py`` from the frozen judgments under
``evaluation/judged/``. That step is deterministic and involves no API calls.

Aggregation, verified against the released tables:

* ``TC`` is the unweighted mean of ``mean_score`` over the ``TC_*`` regimes,
  ``TP`` the same over ``TP_*``; every regime counts equally regardless of n.
* ``balanced`` is ``(TC + TP) / 2``.
* The macro table averages the per-dataset ``TC``/``TP`` with equal weight per
  dataset, not per row, so LongFact's larger n does not dominate.

The script that originally produced the released tables is not in the
repository (see the PR's open questions). This one reproduces every released
value to within 1e-12, but not always bit-for-bit: a handful of cells differ
in the last one or two units in the last place because the original used a
different summation order. Rather than churn published bytes over float
noise, the default mode is **append-only** -- an existing row is copied
through verbatim once its recomputed values are confirmed to match, and only
genuinely new baselines are written fresh. Pass ``--rewrite_all`` to
regenerate every row from scratch instead.

Use ``--verify`` to check the recomputation against the existing tables
without writing anything.
"""

TOLERANCE = 1e-12

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from statistics import mean

# Paper presentation order. Baselines outside this list are appended in sorted
# order so a newly released method still shows up rather than being dropped.
BASELINE_ORDER = [
    "parametric",
    "text_only",
    "full_mm",
    "cascaded_router",
    "field_selector",
    "soft_conductor",
    "selfcheck_gate",
    "answer_consensus",
]

DATASET_DISPLAY = {
    "alpacafact": "AlpacaFact",
    "biography": "Biography",
    "fava": "FAVA",
    "longfact": "LongFact",
}


def load_results(paths):
    """dataset -> baseline -> {regime: mean_score}."""
    out = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        stem = os.path.basename(path)
        key = stem.split("_qimg7_")[0].lower()
        display = DATASET_DISPLAY.get(key, key)
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            sys.exit(f"error: no rows in {path}")
        for row in rows:
            score = row.get("mean_score", "")
            if score == "":
                continue  # regime had no scorable rows; excluded from the mean
            out[display][row["baseline"]][row["regime"]] = float(score)
    return out


def per_dataset_rows(results, order):
    rows = []
    for dataset in sorted(results):
        for baseline in order:
            regimes = results[dataset].get(baseline)
            if not regimes:
                continue
            tc = [s for r, s in regimes.items() if r.startswith("TC_")]
            tp = [s for r, s in regimes.items() if r.startswith("TP_")]
            if not tc or not tp:
                print(f"warning: {dataset}/{baseline} missing TC or TP regimes; "
                      f"skipped", file=sys.stderr)
                continue
            tc_avg, tp_avg = mean(tc), mean(tp)
            rows.append({
                "dataset": dataset,
                "baseline": baseline,
                "TC": tc_avg,
                "TP": tp_avg,
                "balanced": (tc_avg + tp_avg) / 2,
            })
    return rows


def macro_rows(per_dataset, order):
    by_baseline = defaultdict(list)
    for row in per_dataset:
        by_baseline[row["baseline"]].append(row)
    rows = []
    for baseline in order:
        entries = by_baseline.get(baseline)
        if not entries:
            continue
        tc_avg = mean(e["TC"] for e in entries)
        tp_avg = mean(e["TP"] for e in entries)
        rows.append({
            "baseline": baseline,
            "TC_avg": tc_avg,
            "TP_avg": tp_avg,
            "drop": tc_avg - tp_avg,
            "balanced_avg": (tc_avg + tp_avg) / 2,
        })
    return rows


def read_existing(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["baseline"] if "dataset" not in r
                else (r["dataset"], r["baseline"]): r
                for r in csv.DictReader(fh)}


def row_key(row):
    return (row["dataset"], row["baseline"]) if "dataset" in row else row["baseline"]


def write_csv(path, rows, fields, existing=None):
    """Write rows, copying any matching existing row through byte-for-byte."""
    preserved = 0
    out = []
    for row in rows:
        old = (existing or {}).get(row_key(row))
        if old is not None:
            for field in fields:
                if field in ("dataset", "baseline"):
                    continue
                if abs(float(row[field]) - float(old[field])) > TOLERANCE:
                    sys.exit(f"FAIL: {path}: {row_key(row)} {field} recomputed as "
                             f"{row[field]} but the released table says {old[field]} "
                             f"(delta exceeds {TOLERANCE}). Refusing to overwrite a "
                             f"published value; investigate before proceeding.")
            out.append(old)
            preserved += 1
        else:
            out.append(row)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    return preserved, len(out) - preserved


def verify(path, rows, key_fields):
    """Report rows whose values changed against an existing table."""
    if not os.path.exists(path):
        print(f"  {path}: no existing table to compare")
        return True
    with open(path, newline="", encoding="utf-8") as fh:
        old = {tuple(r[k] for k in key_fields): r for r in csv.DictReader(fh)}
    new = {tuple(str(r[k]) for k in key_fields): r for r in rows}
    changed, added = [], []
    for key, row in new.items():
        if key not in old:
            added.append(key)
            continue
        for field, value in row.items():
            if field in key_fields:
                continue
            if abs(float(value) - float(old[key][field])) > 1e-12:
                changed.append((key, field, old[key][field], value))
    removed = [k for k in old if k not in new]
    print(f"  {path}: {len(old)} existing rows, "
          f"{len(added)} added, {len(changed)} changed, {len(removed)} removed")
    for key in added:
        print(f"    + {'/'.join(key)}")
    for key, field, was, now in changed:
        print(f"    ! {'/'.join(key)} {field}: {was} -> {now}")
    for key in removed:
        print(f"    - {'/'.join(key)}")
    return not changed and not removed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_csvs", nargs="+", required=True,
                    help="Per-dataset *_final_results_*.csv files.")
    ap.add_argument("--out_macro", default="evaluation/tables/table_main_qimg7_macro.csv")
    ap.add_argument("--out_per_dataset",
                    default="evaluation/tables/table_per_dataset_qimg7.csv")
    ap.add_argument("--verify", action="store_true",
                    help="Compare against the existing tables and exit non-zero "
                         "if any existing row's values changed. Writes nothing.")
    ap.add_argument("--rewrite_all", action="store_true",
                    help="Regenerate every row instead of preserving the bytes of "
                         "rows already present in the released tables.")
    args = ap.parse_args()

    missing = [p for p in args.results_csvs if not os.path.isfile(p)]
    if missing:
        sys.exit("error: input file(s) not found:\n  " + "\n  ".join(missing))

    results = load_results(args.results_csvs)
    if not results:
        sys.exit("error: no usable rows in the supplied results CSVs")

    seen = {b for ds in results.values() for b in ds}
    order = BASELINE_ORDER + sorted(seen - set(BASELINE_ORDER))
    unknown = seen - set(BASELINE_ORDER)
    if unknown:
        print(f"note: baselines not in the paper order, appended: "
              f"{', '.join(sorted(unknown))}", file=sys.stderr)

    per_ds = per_dataset_rows(results, order)
    macro = macro_rows(per_ds, order)

    if args.verify:
        print("verifying against existing tables:")
        ok = verify(args.out_macro, macro, ["baseline"])
        ok &= verify(args.out_per_dataset, per_ds, ["dataset", "baseline"])
        if not ok:
            sys.exit("FAIL: an existing row's values changed; do not overwrite "
                     "the released tables without investigating")
        print("OK: no existing row changed")
        return

    keep_macro = {} if args.rewrite_all else read_existing(args.out_macro)
    keep_per_ds = {} if args.rewrite_all else read_existing(args.out_per_dataset)

    kept, new = write_csv(args.out_macro, macro,
                          ["baseline", "TC_avg", "TP_avg", "drop", "balanced_avg"],
                          keep_macro)
    print(f"{args.out_macro}: {kept} rows preserved, {new} added")
    kept, new = write_csv(args.out_per_dataset, per_ds,
                          ["dataset", "baseline", "TC", "TP", "balanced"],
                          keep_per_ds)
    print(f"{args.out_per_dataset}: {kept} rows preserved, {new} added")


if __name__ == "__main__":
    main()
