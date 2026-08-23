import argparse
import json
from pathlib import Path

def load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_jsonl", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--kind", choices=["prediction", "judged"], required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    bench = load_jsonl(args.benchmark_jsonl)
    rows = load_jsonl(args.file)

    expected = {r["row_id"] for r in bench}
    seen = {}
    duplicates = 0
    bad = 0

    for r in rows:
        rid = r.get("row_id")
        if rid in seen:
            duplicates += 1
        seen[rid] = r

    missing = expected - set(seen)
    extra = set(seen) - expected

    for rid, r in seen.items():
        if args.kind == "prediction":
            if r.get("error") or not str(r.get("prediction", "")).strip():
                bad += 1
        else:
            if r.get("error") or r.get("score") is None:
                bad += 1

    print("file:", args.file)
    print("kind:", args.kind)
    print("expected:", len(expected))
    print("rows:", len(rows))
    print("unique_row_ids:", len(seen))
    print("duplicates:", duplicates)
    print("missing:", len(missing))
    print("extra:", len(extra))
    print("bad_latest_rows:", bad)

    if args.strict and (missing or bad or extra):
        raise SystemExit("VALIDATION FAILED")

if __name__ == "__main__":
    main()
