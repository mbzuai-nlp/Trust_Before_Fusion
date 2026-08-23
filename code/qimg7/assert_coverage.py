import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_jsonl", required=True)
    ap.add_argument("--expected_questions", type=int, required=True)
    ap.add_argument("--attacks", nargs="+", required=True)
    ap.add_argument("--min_attack_frac", type=float, default=0.8)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    rows = []
    with open(args.benchmark_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    qids = set(r["qid"] for r in rows)
    counts = Counter(r["regime"] for r in rows)

    print("benchmark:", args.benchmark_jsonl)
    print("rows:", len(rows))
    print("unique_questions:", len(qids))
    print("expected_questions:", args.expected_questions)
    print("regime_counts:", dict(counts))

    ok = True

    if len(qids) < args.expected_questions:
        print(f"[FAIL] only {len(qids)} questions, expected {args.expected_questions}")
        ok = False

    required = ["TC_IC", "TP_IC"]
    for attack in args.attacks:
        required.append(f"TC_IP_{attack}")
        required.append(f"TP_IP_{attack}")

    min_attack = max(1, int(args.expected_questions * args.min_attack_frac))

    for reg in required:
        c = counts.get(reg, 0)
        threshold = args.expected_questions if reg in ["TC_IC", "TP_IC"] else min_attack
        if c < threshold:
            print(f"[FAIL] {reg}: {c}, expected at least {threshold}")
            ok = False

    if args.strict and not ok:
        raise SystemExit("COVERAGE CHECK FAILED")

    if ok:
        print("COVERAGE CHECK PASSED")

if __name__ == "__main__":
    main()
