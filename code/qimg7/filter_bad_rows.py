import argparse
import json
from pathlib import Path

def is_bad(r, kind):
    if kind == "prediction":
        return bool(r.get("error")) or not str(r.get("prediction", "")).strip()
    if kind == "judged":
        return bool(r.get("error")) or r.get("score") is None
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--kind", choices=["prediction", "judged"], required=True)
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        print("No file to filter:", p)
        return

    latest = {}
    order = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = r.get("row_id")
            if rid not in latest:
                order.append(rid)
            latest[rid] = r

    kept = []
    bad = 0
    for rid in order:
        r = latest[rid]
        if is_bad(r, args.kind):
            bad += 1
        else:
            kept.append(r)

    backup = str(p) + ".bak"
    p.rename(backup)
    with open(p, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("input:", args.input)
    print("backup:", backup)
    print("kept:", len(kept))
    print("removed_bad_or_duplicate_latest:", bad)

if __name__ == "__main__":
    main()
