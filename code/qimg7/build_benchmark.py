import argparse
import ast
import json
import re
from pathlib import Path
import pandas as pd
from collections import Counter

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s

def parse_evidence_cell(x, top_k=5):
    if pd.isna(x):
        return []
    s = str(x)
    try:
        obj = ast.literal_eval(s)
        items = obj if isinstance(obj, list) else [s]
    except Exception:
        items = [s]

    out = []
    for item in items[:top_k]:
        item = str(item)
        parts = item.split("|")
        if len(parts) >= 3:
            title = parts[0].strip()
            snippet = "|".join(parts[1:-1]).strip()
            url = parts[-1].strip()
        elif len(parts) == 2:
            title = parts[0].strip()
            snippet = parts[1].strip()
            url = ""
        else:
            title = ""
            snippet = item.strip()
            url = ""
        out.append({"title": title, "snippet": snippet, "url": url})
    return out

def load_text_rows(path, top_k=5):
    df = pd.read_csv(path)
    if "question" not in df.columns:
        raise ValueError(f"No question column in {path}")
    if "total_evidence" not in df.columns:
        raise ValueError(f"No total_evidence column in {path}")

    rows = []
    for i, r in df.iterrows():
        rows.append({
            "idx": i,
            "question": str(r["question"]),
            "text_evidence": parse_evidence_cell(r["total_evidence"], top_k=top_k),
        })

    by_norm_q = {norm(r["question"]): r for r in rows}
    return rows, by_norm_q

def image_usable(img):
    return bool(
        img.get("resolved_local_path")
        or img.get("local_path")
        or img.get("image_url")
        or img.get("url")
    )

def pick(items):
    for x in items:
        if image_usable(x):
            return x
    return items[0] if items else None

def resolve_text_row(image_row, clean_rows, clean_by_norm, polluted_rows, polluted_by_norm, index_base):
    qidx = int(image_row.get("question_idx", -1))
    img_q = image_row["question"]

    if index_base == "auto":
        candidates = [qidx, qidx - 1]
    elif index_base == "0":
        candidates = [qidx]
    elif index_base == "1":
        candidates = [qidx - 1]
    else:
        raise ValueError("index_base must be auto, 0, or 1")

    for idx in candidates:
        if 0 <= idx < len(clean_rows) and 0 <= idx < len(polluted_rows):
                                                                      
            return clean_rows[idx], polluted_rows[idx], "index"

    nq = norm(img_q)
    if nq in clean_by_norm and nq in polluted_by_norm:
        return clean_by_norm[nq], polluted_by_norm[nq], "question"

    return None, None, "missing"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_jsonl", required=True)
    ap.add_argument("--clean_text_csv", required=True)
    ap.add_argument("--polluted_text_csv", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--top_k_text", type=int, default=5)
    ap.add_argument("--max_questions", type=int, default=0)
    ap.add_argument("--image_pollution_types", nargs="+", required=True)
    ap.add_argument("--index_base", choices=["auto", "0", "1"], default="auto")
    args = ap.parse_args()

    clean_rows, clean_by_norm = load_text_rows(args.clean_text_csv, args.top_k_text)
    polluted_rows, polluted_by_norm = load_text_rows(args.polluted_text_csv, args.top_k_text)

    image_rows = []
    with open(args.image_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            image_rows.append(json.loads(line))

    out = []
    row_id = 0
    qid = 0
    align_counts = Counter()
    skipped = 0

    for image_row in image_rows:
        clean_text_row, polluted_text_row, align_mode = resolve_text_row(
            image_row,
            clean_rows,
            clean_by_norm,
            polluted_rows,
            polluted_by_norm,
            args.index_base,
        )
        align_counts[align_mode] += 1

        if clean_text_row is None or polluted_text_row is None:
            skipped += 1
            continue

        q = clean_text_row["question"]
        clean_img = pick(image_row.get("clean", []))
        if not clean_img:
            skipped += 1
            continue

        def add(regime, text_ev, img_ev, text_status, image_status, attack):
            nonlocal row_id
            out.append({
                "row_id": row_id,
                "qid": qid,
                "dataset": args.dataset,
                "question": q,
                "original_image_question": image_row.get("question", ""),
                "question_idx": image_row.get("question_idx", None),
                "text_idx": clean_text_row["idx"],
                "regime": regime,
                "text_status": text_status,
                "image_status": image_status,
                "image_pollution_type": attack,
                "text_evidence": text_ev,
                "image_evidence": img_ev,
            })
            row_id += 1

        add("TC_IC", clean_text_row["text_evidence"], clean_img, "clean", "clean", "clean")
        add("TP_IC", polluted_text_row["text_evidence"], clean_img, "polluted", "clean", "clean")

        for attack in args.image_pollution_types:
            img = pick(image_row.get("polluted", {}).get(attack, []))
            if not img:
                continue
            add(f"TC_IP_{attack}", clean_text_row["text_evidence"], img, "clean", "polluted", attack)
            add(f"TP_IP_{attack}", polluted_text_row["text_evidence"], img, "polluted", "polluted", attack)

        qid += 1
        if args.max_questions and qid >= args.max_questions:
            break

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter(r["regime"] for r in out)
    print(f"Wrote {len(out)} rows -> {args.output_jsonl}")
    print("unique_questions:", len(set(r["qid"] for r in out)))
    print("alignment_counts:", dict(align_counts))
    print("skipped:", skipped)
    print("Regime counts:", dict(counts))

if __name__ == "__main__":
    main()
