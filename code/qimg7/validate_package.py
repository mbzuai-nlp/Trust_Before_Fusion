import argparse
import os
from pathlib import Path
import pandas as pd
from collections import Counter

DATASETS = ["LongFact", "Biography", "AlpacaFact", "FAVA"]

ALL_ATTACKS = [
    "caption_flip",
    "entity_swap",
    "semantic_entity_rewrite",
    "figstep_typography",
    "adversarial_patch",
    "image_blend",
    "neural_style_transfer",
]

LOCAL_REQUIRED = {
    "semantic_entity_rewrite",
    "figstep_typography",
    "adversarial_patch",
    "image_blend",
    "neural_style_transfer",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package_dir", required=True)
    ap.add_argument("--strict_local_paths", action="store_true")
    args = ap.parse_args()

    pkg = Path(args.package_dir)
    if not pkg.exists():
        raise SystemExit(f"Package dir not found: {pkg}")

    ok = True
    print("PACKAGE:", pkg.resolve())

    for ds in DATASETS:
        csv_path = pkg / f"{ds}_query_image_polluted.csv"
        if not csv_path.exists():
            print(f"[MISSING] {csv_path}")
            ok = False
            continue

        df = pd.read_csv(csv_path)
        print(f"\n== {ds} ==")
        print("rows:", len(df))
        print("questions:", df["question"].nunique())
        print("columns:", list(df.columns))

        vc = df["pollution_type"].value_counts().to_dict()
        print("attack_counts:", vc)

        missing_attacks = [a for a in ALL_ATTACKS if a not in vc]
        if missing_attacks:
            print("[WARN] missing attacks:", missing_attacks)

        local_df = df[df["pollution_type"].isin(LOCAL_REQUIRED)].copy()
        missing_paths = []
        existing_paths = 0

        for _, r in local_df.iterrows():
            rel = r.get("polluted_image_path")
            if pd.isna(rel) or not str(rel).strip():
                missing_paths.append(("EMPTY", r["pollution_type"], r["question_idx"], r["image_idx"]))
                continue
            p = pkg / str(rel)
            if p.exists():
                existing_paths += 1
            else:
                missing_paths.append((str(rel), r["pollution_type"], r["question_idx"], r["image_idx"]))

        print("local_attack_rows:", len(local_df))
        print("local_paths_existing:", existing_paths)
        print("local_paths_missing:", len(missing_paths))

        if missing_paths[:5]:
            print("missing_path_examples:")
            for x in missing_paths[:5]:
                print(" ", x)

        if args.strict_local_paths and missing_paths:
            ok = False

    if not ok:
        raise SystemExit("\nVALIDATION FAILED. Fix missing CSVs/images before running full experiments.")

    print("\nVALIDATION PASSED.")

if __name__ == "__main__":
    main()
