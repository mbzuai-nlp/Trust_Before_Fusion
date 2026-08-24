#!/usr/bin/env python3
"""
Compute inter-annotator agreement (Cohen's kappa) between two validation CSVs.

Usage:
    python compute_iaa.py annotator1.csv annotator2.csv [output.txt]

Both CSVs must have a `row_id` column and the annotation fields below.
Computes kappa + raw agreement % for each field, overall and broken down by
pollution_type and dataset. If one rater's column is constant, kappa is
undefined (NaN) and the row is flagged explicitly instead of being printed
as "0.000" or "nan".
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42

FIELDS = ["on_topic", "fact_flipped", "plausible"]
FIELD_LABELS = {
    "on_topic":     "On-topic",
    "fact_flipped": "Fact flipped",
    "plausible":    "Visually plausible",
}

_OUT_FH = None


def _emit(line: str = "") -> None:
    print(line)
    if _OUT_FH is not None:
        _OUT_FH.write(line + "\n")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = ({"row_id"} | set(FIELDS)) - set(df.columns)
    if missing:
        sys.exit(f"Error: {path} missing columns: {missing}")
    return df.set_index("row_id")


def kappa_label(k: float) -> str:
    if k < 0:     return "poor (< chance)"
    if k < 0.20:  return "slight"
    if k < 0.40:  return "fair"
    if k < 0.60:  return "moderate"
    if k < 0.80:  return "substantial"
    return "almost perfect"


def _bootstrap_kappa_ci(v1, v2, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED):
    """Percentile 95% CI for Cohen's kappa via paired bootstrap.

    Returns (lo, hi, n_valid_resamples). Resamples where either rater is
    constant produce undefined kappa and are dropped from the percentile.
    """
    rng = np.random.default_rng(seed)
    n = len(v1)
    a1 = np.asarray(v1)
    a2 = np.asarray(v2)
    ks = np.empty(B, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            r1 = a1[idx]; r2 = a2[idx]
            if len(set(r1)) < 2 or len(set(r2)) < 2:
                ks[b] = np.nan
                continue
            ks[b] = cohen_kappa_score(r1, r2)
    ks = ks[~np.isnan(ks)]
    if len(ks) < 2:
        return None, None, len(ks)
    lo, hi = np.percentile(ks, [2.5, 97.5])
    return float(lo), float(hi), len(ks)


def compute_field(s1: pd.Series, s2: pd.Series, with_ci: bool = True):
    """Return (kappa, agree_pct, n, note, ci) for two aligned series.

    kappa is None when undefined (one rater constant, or n<2).
    ci is (lo, hi) or None.
    note explains why kappa is None when applicable.
    """
    s1 = s1.astype(str).str.strip().str.lower()
    s2 = s2.astype(str).str.strip().str.lower()
    valid = s1.notna() & s2.notna() & (s1 != "") & (s1 != "nan") & (s2 != "") & (s2 != "nan")
    n = int(valid.sum())
    if n < 2:
        return None, None, n, "n<2", None
    v1, v2 = s1[valid].tolist(), s2[valid].tolist()
    agree = sum(x == y for x, y in zip(v1, v2)) / n * 100
    # Constant-rater check: kappa is undefined if either rater has no variance.
    if len(set(v1)) < 2 or len(set(v2)) < 2:
        which = []
        if len(set(v1)) < 2: which.append(f"A1 constant='{v1[0]}'")
        if len(set(v2)) < 2: which.append(f"A2 constant='{v2[0]}'")
        return None, agree, n, "kappa undefined (" + ", ".join(which) + ")", None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = cohen_kappa_score(v1, v2)
    if pd.isna(k):
        return None, agree, n, "kappa undefined (degenerate marginals)", None
    ci = None
    if with_ci and n >= 5:
        lo, hi, _ = _bootstrap_kappa_ci(v1, v2)
        if lo is not None:
            ci = (lo, hi)
    return float(k), agree, n, "", ci


def print_table(rows, indent: str = "") -> None:
    """rows: list of (label, kappa, agree_pct, n, note, ci)"""
    hdr = f"{'Field':<28}  {'Kappa':>7}  {'95% CI':>16}  {'Agree':>7}  {'n':>5}  Interpretation"
    _emit(indent + hdr)
    _emit(indent + "-" * len(hdr))
    for row in rows:
        label, k, agree, n, note = row[:5]
        ci = row[5] if len(row) > 5 else None
        if k is None:
            ag = f"{agree:>6.1f}%" if agree is not None else f"{'N/A':>7}"
            _emit(f"{indent}{label:<28}  {'N/A':>7}  {'N/A':>16}  {ag}  {n:>5}  {note}")
        else:
            ci_str = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci is not None else f"{'n<5':>16}"
            _emit(f"{indent}{label:<28}  {k:>7.3f}  {ci_str:>16}  {agree:>6.1f}%  {n:>5}  {kappa_label(k)}")


def print_distribution(a1s: pd.DataFrame, a2s: pd.DataFrame) -> None:
    _emit(f"\n{'─'*60}")
    _emit("LABEL DISTRIBUTIONS (sanity check for constant-rater issues)")
    _emit(f"{'─'*60}")
    for f in FIELDS:
        c1 = a1s[f].astype(str).str.strip().str.lower().value_counts().to_dict()
        c2 = a2s[f].astype(str).str.strip().str.lower().value_counts().to_dict()
        _emit(f"  {FIELD_LABELS[f]}")
        _emit(f"    A1: {c1}")
        _emit(f"    A2: {c2}")


def main():
    global _OUT_FH
    if len(sys.argv) not in (3, 4):
        sys.exit("Usage: python compute_iaa.py annotator1.csv annotator2.csv [output.txt]")

    out_path = sys.argv[3] if len(sys.argv) == 4 else "iaa_results.txt"
    _OUT_FH = open(out_path, "w", encoding="utf-8")

    try:
        a1 = load(sys.argv[1])
        a2 = load(sys.argv[2])

        shared = a1.index.intersection(a2.index)
        _emit(f"\nAnnotator 1: {sys.argv[1]}  ({len(a1)} rows)")
        _emit(f"Annotator 2: {sys.argv[2]}  ({len(a2)} rows)")
        _emit(f"Shared row_ids: {len(shared)}  |  only in A1: {len(a1)-len(shared)}  |  only in A2: {len(a2)-len(shared)}")

        if len(shared) == 0:
            sys.exit("\nNo shared row_ids — cannot compute agreement.")

        a1s, a2s = a1.loc[shared], a2.loc[shared]

        print_distribution(a1s, a2s)

        # ── Overall ──────────────────────────────────────────────────────────
        _emit(f"\n{'─'*60}")
        _emit("OVERALL AGREEMENT")
        _emit(f"{'─'*60}")
        rows = []
        for f in FIELDS:
            k, ag, n, note, ci = compute_field(a1s[f], a2s[f])
            rows.append((FIELD_LABELS[f], k, ag, n, note, ci))
        print_table(rows)

        # ── By pollution type ────────────────────────────────────────────────
        if "pollution_type" in a1s.columns:
            _emit(f"\n{'─'*60}")
            _emit("BY POLLUTION TYPE")
            for pt in sorted(a1s["pollution_type"].dropna().unique()):
                idx = a1s.index[a1s["pollution_type"] == pt]
                _emit(f"\n  {pt}  (n={len(idx)})")
                rows = []
                for f in FIELDS:
                    k, ag, n, note, ci = compute_field(a1s.loc[idx, f], a2s.loc[idx, f])
                    rows.append((f"  {FIELD_LABELS[f]}", k, ag, n, note, ci))
                print_table(rows, indent="  ")

        # ── By dataset ───────────────────────────────────────────────────────
        if "dataset" in a1s.columns:
            _emit(f"\n{'─'*60}")
            _emit("BY DATASET")
            for ds in sorted(a1s["dataset"].dropna().unique()):
                idx = a1s.index[a1s["dataset"] == ds]
                _emit(f"\n  {ds}  (n={len(idx)})")
                rows = []
                for f in FIELDS:
                    k, ag, n, note, ci = compute_field(a1s.loc[idx, f], a2s.loc[idx, f])
                    rows.append((f"  {FIELD_LABELS[f]}", k, ag, n, note, ci))
                print_table(rows, indent="  ")

        # ── Per-item disagreements ───────────────────────────────────────────
        _emit(f"\n{'─'*60}")
        _emit("DISAGREEMENTS (any field differs)")
        disagree_ids = []
        for rid in shared:
            for f in FIELDS:
                v1 = str(a1s.loc[rid, f] or "").strip().lower()
                v2 = str(a2s.loc[rid, f] or "").strip().lower()
                if v1 and v2 and v1 != v2:
                    disagree_ids.append(rid)
                    break
        _emit(f"  {len(disagree_ids)} / {len(shared)} items have at least one disagreement")
        if disagree_ids:
            _emit(f"\n  {'row_id':<45} {'Field':<18} A1      A2")
            _emit(f"  {'-'*80}")
            for rid in disagree_ids[:30]:
                for f in FIELDS:
                    v1 = str(a1s.loc[rid, f] or "").strip().lower()
                    v2 = str(a2s.loc[rid, f] or "").strip().lower()
                    if v1 and v2 and v1 != v2:
                        _emit(f"  {rid:<45} {FIELD_LABELS[f]:<18} {v1:<7} {v2}")
            if len(disagree_ids) > 30:
                _emit(f"  … and {len(disagree_ids)-30} more")
        _emit()
    finally:
        if _OUT_FH is not None:
            _OUT_FH.close()
            print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
