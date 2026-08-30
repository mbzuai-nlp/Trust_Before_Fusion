#!/usr/bin/env python3
"""Check the released artifacts against each other. No API calls, no network.

Four checks:

1. Every released prediction and judgment validates against the benchmark it
   was produced for (`code/qimg7/validate_outputs.py --strict`).
2. Regenerating each per-dataset results CSV from the frozen judgments
   reproduces the committed file (line endings normalised).
3. Regenerating the aggregate tables reproduces every released row to within
   1e-12 (`code/eval/make_paper_tables.py --verify`).
4. The image tree reconciles against the pool CSVs and the manifest.

Exit status is 0 only if every check passes.
"""

import argparse
import glob
import hashlib
import os
import subprocess
import sys
import tempfile

DATASETS = [("alpacafact", 20), ("biography", 20), ("fava", 20), ("longfact", 50)]
BASELINES = [
    "parametric", "text_only", "full_mm", "selfcheck_gate", "cascaded_router",
    "source_aware_conductor", "source_aware_selector", "field_selector",
    "soft_conductor", "answer_consensus",
]


class Report:
    def __init__(self):
        self.failures = []
        self.passes = 0

    def check(self, ok, label, detail=""):
        if ok:
            self.passes += 1
            print(f"  ok    {label}")
        else:
            self.failures.append(f"{label}{': ' + detail if detail else ''}")
            print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def normalised(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_tag", default="gpt4omini")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    py, tag = args.python, args.model_tag
    rep = Report()

    print("\n[1/4] released predictions and judgments validate against the benchmark")
    for dataset, n in DATASETS:
        bench = f"benchmark/evaluated/{dataset}_qimg7_mm_regimes_{n}.jsonl"
        if not os.path.exists(bench):
            rep.check(False, f"{dataset}: benchmark present", bench)
            continue
        for baseline in BASELINES:
            for kind, path in (
                ("prediction", f"evaluation/predictions/{tag}/{dataset}/{baseline}_{n}.jsonl"),
                ("judged", f"evaluation/judged/{tag}/{dataset}/{baseline}_{n}.judged.jsonl"),
            ):
                if not os.path.exists(path):
                    continue  # not every baseline has both; absence is REP-03, not a mismatch
                res = run([py, "code/qimg7/validate_outputs.py",
                           "--benchmark_jsonl", bench, "--file", path,
                           "--kind", kind, "--strict"])
                rep.check(res.returncode == 0, f"{dataset}/{baseline} {kind}",
                          (res.stderr or res.stdout).strip()[-300:])

    print("\n[2/4] per-dataset results regenerate from the frozen judgments")
    with tempfile.TemporaryDirectory() as tmp:
        for dataset, n in DATASETS:
            judged = sorted(glob.glob(f"evaluation/judged/{tag}/{dataset}/*.judged.jsonl"))
            committed = f"evaluation/results/{dataset}/{dataset}_qimg7_final_results_{tag}_{n}.csv"
            if not judged or not os.path.exists(committed):
                rep.check(False, f"{dataset}: inputs present", committed)
                continue
            out = os.path.join(tmp, f"{dataset}.csv")
            res = run([py, "code/eval/make_results_table.py", "--judged_jsonls", *judged,
                       "--output_csv", out, "--output_md", out.replace(".csv", ".md")])
            if res.returncode != 0:
                rep.check(False, f"{dataset}: regeneration ran",
                          (res.stderr or res.stdout).strip()[-300:])
                continue
            rep.check(normalised(out) == normalised(committed),
                      f"{dataset}: regenerated results match the committed CSV",
                      "content differs from the released file")

    print("\n[3/4] aggregate tables reproduce every released row")
    res = run([py, "code/eval/make_paper_tables.py", "--results_csvs",
               *sorted(glob.glob(f"evaluation/results/*/*_final_results_{tag}_*.csv")),
               "--verify"])
    rep.check(res.returncode == 0, "aggregate tables verify",
              (res.stdout + res.stderr).strip()[-500:])

    print("\n[4/4] image tree reconciles against the pool CSVs")
    res = run([py, "code/qimg7/build_image_hash_manifest.py", "--counts_only"])
    rep.check(res.returncode == 0, "image manifest reconciliation ran",
              (res.stderr or "").strip()[-300:])
    if res.returncode == 0:
        counts = dict()
        for line in res.stdout.splitlines():
            if line.strip().startswith(("pool image paths missing", "role=unreferenced")):
                counts[line.rsplit(None, 1)[0].strip()] = line.rsplit(None, 1)[1]
        for label, value in counts.items():
            rep.check(value == "0", f"{label} == 0", f"got {value}")
        print(res.stdout.rstrip())

    print(f"\n{rep.passes} checks passed, {len(rep.failures)} failed")
    if rep.failures:
        print("\nFailures:")
        for failure in rep.failures:
            print(f"  - {failure}")
        return 1
    print("All released artifacts verified. No API calls were made.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
