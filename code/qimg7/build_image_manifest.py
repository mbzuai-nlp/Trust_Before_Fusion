import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

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

def str_or_empty(x):
    if pd.isna(x):
        return ""
    return str(x)

def img_obj_from_row(r, pkg, clean=False):
    if clean:
        url = str_or_empty(r["original_image_url"])
        alt = str_or_empty(r["original_alt"])
        path = ""
        ptype = "clean"
    else:
        url = str_or_empty(r["polluted_image_url"])
        alt = str_or_empty(r["polluted_alt"])
        ptype = str_or_empty(r["pollution_type"])
        rel = str_or_empty(r.get("polluted_image_path", ""))
        if rel:
            full = pkg / rel
            path = str(full.resolve()) if full.exists() else str(full)
        else:
            path = ""

    return {
        "image_url": url,
        "url": url,
        "alt_text": alt,
        "title": alt,
        "page_url": "",
        "resolved_local_path": path,
        "local_path": path,
        "pollution_type": ptype,
        "metadata": {
            "question_idx": int(r["question_idx"]),
            "image_idx": int(r["image_idx"]),
            "manipulation_method": str_or_empty(r.get("manipulation_method", "")),
            "manipulation_prompt": str_or_empty(r.get("manipulation_prompt", "")),
            "donor_question_idx": str_or_empty(r.get("donor_question_idx", "")),
            "rationale": str_or_empty(r.get("rationale", "")),
            "original_alt": str_or_empty(r.get("original_alt", "")),
            "polluted_alt": str_or_empty(r.get("polluted_alt", "")),
        }
    }

def usable_polluted(r, pkg, require_local=True):
    ptype = str_or_empty(r["pollution_type"])
    if ptype in LOCAL_REQUIRED:
        rel = str_or_empty(r.get("polluted_image_path", ""))
        if not rel:
            return False
        if require_local and not (pkg / rel).exists():
            return False
        return True
    return bool(str_or_empty(r["polluted_image_url"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--attacks", nargs="+", default=ALL_ATTACKS)
    ap.add_argument("--max_clean_per_question", type=int, default=5)
    ap.add_argument("--max_polluted_per_attack", type=int, default=3)
    ap.add_argument("--allow_missing_local", action="store_true")
    args = ap.parse_args()

    pkg = Path(args.package_dir)
    csv_path = pkg / f"{args.dataset}_query_image_polluted.csv"
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    attacks = set(args.attacks)

    out_rows = []
    skipped_missing_local = 0

    for qidx, g in df.groupby("question_idx"):
        question = str(g["question"].iloc[0])

        clean_by_img = {}
        for _, r in g.iterrows():
            img_idx = int(r["image_idx"])
            if img_idx not in clean_by_img:
                clean_by_img[img_idx] = img_obj_from_row(r, pkg, clean=True)

        clean = [clean_by_img[k] for k in sorted(clean_by_img)[:args.max_clean_per_question]]

        polluted = defaultdict(list)
        for _, r in g.iterrows():
            ptype = str_or_empty(r["pollution_type"])
            if ptype not in attacks:
                continue
            if not usable_polluted(r, pkg, require_local=(not args.allow_missing_local)):
                skipped_missing_local += 1
                continue
            if len(polluted[ptype]) >= args.max_polluted_per_attack:
                continue
            polluted[ptype].append(img_obj_from_row(r, pkg, clean=False))

        out_rows.append({
            "dataset": args.dataset,
            "question_idx": int(qidx),
            "question": question,
            "clean": clean,
            "polluted": dict(polluted),
            "counts": {
                "clean": len(clean),
                **{k: len(v) for k, v in polluted.items()}
            }
        })

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter
    attack_counts = Counter()
    q_coverage = Counter()
    for row in out_rows:
        for k, v in row["polluted"].items():
            attack_counts[k] += len(v)
            if len(v) > 0:
                q_coverage[k] += 1

    print(f"Wrote {len(out_rows)} question rows -> {args.output_jsonl}")
    print("attack_image_counts:", dict(attack_counts))
    print("attack_question_coverage:", dict(q_coverage))
    print("skipped_missing_local_or_unusable:", skipped_missing_local)

    if skipped_missing_local and not args.allow_missing_local:
        raise SystemExit("Missing local polluted images were skipped. Check package completeness.")

if __name__ == "__main__":
    main()
