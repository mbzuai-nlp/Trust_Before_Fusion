#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


SOURCES = ("parametric", "text_only", "full_mm")


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: str) -> dict:
    return {obj["row_id"]: obj for obj in iter_jsonl(path)}


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(a: str, b: str) -> float:
    ta = normalize(a).split()
    tb = normalize(b).split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    overlap = sum((ca & cb).values())
    return (2.0 * overlap) / (len(ta) + len(tb))


def seq_ratio(a: str, b: str) -> float:
    na = normalize(a)
    nb = normalize(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def similarity(a: str, b: str) -> float:
    return 0.7 * token_f1(a, b) + 0.3 * seq_ratio(a, b)


def model_name_from_prediction(pred_obj: dict) -> str:
    return pred_obj.get("prediction_model", pred_obj.get("model", ""))


def choose_consensus(candidates: dict, threshold: float, tie_break: str) -> tuple[str, dict]:
    sims = {}
    for i, src_a in enumerate(SOURCES):
        for src_b in SOURCES[i + 1 :]:
            sims[f"{src_a}__{src_b}"] = similarity(
                candidates[src_a].get("prediction", ""),
                candidates[src_b].get("prediction", ""),
            )

    avg = {}
    for src in SOURCES:
        vals = []
        for other in SOURCES:
            if src == other:
                continue
            key = "__".join(sorted([src, other], key=SOURCES.index))
            vals.append(sims[key])
        avg[src] = sum(vals) / len(vals)

    max_pair_key, max_pair_score = max(sims.items(), key=lambda kv: kv[1])
    pair_sources = max_pair_key.split("__")

                                                                              
                                                                        
    pool = pair_sources if max_pair_score >= threshold else list(SOURCES)
    ranked = sorted(
        pool,
        key=lambda src: (
            avg[src],
            1 if src == tie_break else 0,
            len(str(candidates[src].get("prediction", ""))),
        ),
        reverse=True,
    )
    chosen = ranked[0]

    info = {
        "chosen_source": chosen,
        "pairwise_similarity": sims,
        "average_similarity": avg,
        "max_pair": max_pair_key,
        "max_pair_similarity": max_pair_score,
        "consensus_pair_used": max_pair_score >= threshold,
        "threshold": threshold,
        "tie_break": tie_break,
    }
    return chosen, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_jsonl", required=True)
    ap.add_argument("--param_preds", required=True)
    ap.add_argument("--text_preds", required=True)
    ap.add_argument("--full_preds", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--choices_jsonl", required=True)
    ap.add_argument("--baseline_name", default="answer_consensus")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument(
        "--tie_break",
        choices=SOURCES,
        default="text_only",
        help="Preferred source if centrality scores tie exactly.",
    )
    args = ap.parse_args()

    bench = load_jsonl(args.benchmark_jsonl)
    rows = {
        "parametric": load_jsonl(args.param_preds),
        "text_only": load_jsonl(args.text_preds),
        "full_mm": load_jsonl(args.full_preds),
    }

    out_path = Path(args.output_jsonl)
    choices_path = Path(args.choices_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    choices_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as fout, choices_path.open("w", encoding="utf-8") as fchoices:
        for rid, brow in bench.items():
            if any(rid not in rows[src] for src in SOURCES):
                continue
            candidates = {src: rows[src][rid] for src in SOURCES}
            chosen_source, info = choose_consensus(candidates, args.threshold, args.tie_break)
            chosen = candidates[chosen_source]

            pred_row = {
                "row_id": rid,
                "qid": brow.get("qid", chosen.get("qid")),
                "question": brow.get("question", chosen.get("question")),
                "regime": brow.get("regime", chosen.get("regime")),
                "baseline": args.baseline_name,
                "prediction_model": model_name_from_prediction(chosen),
                "prediction": chosen.get("prediction", ""),
                "error": chosen.get("error", ""),
            }
            choice_row = {
                "row_id": rid,
                "qid": brow.get("qid"),
                "regime": brow.get("regime"),
                "baseline": args.baseline_name,
                **info,
            }

            fout.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
            fchoices.write(json.dumps(choice_row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} consensus predictions -> {out_path}")
    print(f"Wrote consensus choices -> {choices_path}")


if __name__ == "__main__":
    main()
