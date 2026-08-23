import argparse
import json
from pathlib import Path

BAD_TEXT = {"conflicting", "suspicious", "weak", "low", "unreliable"}
GOOD_TEXT = {"trustworthy", "high", "medium", "reliable"}

def norm(x):
    return str(x or "").strip().lower()

def load_jsonl(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows[obj["row_id"]] = obj
    return rows

def choose_row(rid, policy, bench, selfcheck, param, resolver, resolver_choices):
    sc = selfcheck.get(rid, {})
    sc_decision = str(sc.get("decision", "")).upper()

    rc = resolver_choices.get(rid, {})
    text_rel = norm(rc.get("text_reliability"))
    img_rel = norm(rc.get("image_reliability"))
    final_decision = str(rc.get("decision", rc.get("final_decision", ""))).upper()
    chosen_source = norm(rc.get("chosen_source"))

                                                                
    if policy == "strict":
        use_resolver = sc_decision != "FALLBACK"

                                                                                          
    elif policy == "soft":
        if sc_decision != "FALLBACK":
            use_resolver = True
        else:
            use_resolver = (text_rel in GOOD_TEXT and final_decision not in {"FALLBACK", ""})

                                                      
    elif policy == "field":
        use_resolver = not (text_rel in BAD_TEXT or final_decision in {"FALLBACK", ""})

                                                                                  
    elif policy == "conservative":
        use_resolver = (
            sc_decision != "FALLBACK"
            and text_rel not in BAD_TEXT
            and final_decision not in {"FALLBACK", ""}
        )

    else:
        raise ValueError(f"Unknown policy: {policy}")

    if use_resolver:
        src = resolver[rid]
        pred = src.get("prediction", "")
        selected = chosen_source or "resolver"
        decision = final_decision or "RESOLVER"
    else:
        src = param[rid]
        pred = src.get("prediction", "")
        selected = "parametric"
        decision = "FALLBACK"

    pred_row = {
        "row_id": rid,
        "qid": bench[rid].get("qid"),
        "question": bench[rid].get("question"),
        "regime": bench[rid].get("regime"),
        "baseline": "",
        "prediction_model": src.get("prediction_model", ""),
        "prediction": pred,
        "error": "",
    }

    choice_row = {
        "row_id": rid,
        "qid": bench[rid].get("qid"),
        "regime": bench[rid].get("regime"),
        "policy": policy,
        "selfcheck_decision": sc_decision,
        "resolver_decision": final_decision,
        "chosen_source": selected,
        "final_decision": decision,
        "text_reliability": text_rel,
        "image_reliability": img_rel,
    }

    return pred_row, choice_row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_jsonl", required=True)
    ap.add_argument("--selfcheck_choices", required=True)
    ap.add_argument("--param_preds", required=True)
    ap.add_argument("--resolver_preds", required=True)
    ap.add_argument("--resolver_choices", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--choices_jsonl", required=True)
    ap.add_argument("--baseline_name", required=True)
    ap.add_argument("--policy", choices=["strict", "soft", "field", "conservative"], required=True)
    args = ap.parse_args()

    bench = load_jsonl(args.benchmark_jsonl)
    selfcheck = load_jsonl(args.selfcheck_choices)
    param = load_jsonl(args.param_preds)
    resolver = load_jsonl(args.resolver_preds)
    resolver_choices = load_jsonl(args.resolver_choices)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.choices_jsonl).parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout, open(args.choices_jsonl, "w", encoding="utf-8") as fchoices:
        for rid in bench:
            if rid not in param or rid not in resolver:
                continue
            pred_row, choice_row = choose_row(
                rid,
                args.policy,
                bench,
                selfcheck,
                param,
                resolver,
                resolver_choices,
            )
            pred_row["baseline"] = args.baseline_name
            fout.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
            fchoices.write(json.dumps(choice_row, ensure_ascii=False) + "\n")
            n += 1

    print("Wrote", n, "predictions ->", args.output_jsonl)
    print("Wrote choices ->", args.choices_jsonl)

if __name__ == "__main__":
    main()
