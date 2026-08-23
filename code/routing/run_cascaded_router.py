import json
import argparse
from pathlib import Path

def load_jsonl(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows[obj["row_id"]] = obj
    return rows


def model_name_from_prediction(pred_obj):
                                                         
    return pred_obj.get("prediction_model", pred_obj.get("model", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_jsonl", required=True)
    ap.add_argument("--selfcheck_choices", required=True)
    ap.add_argument("--triage_choices", required=True)
    ap.add_argument("--full_preds", required=True)
    ap.add_argument("--text_preds", required=True)
    ap.add_argument("--param_preds", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--choices_jsonl", required=True)
    args = ap.parse_args()

    bench = load_jsonl(args.benchmark_jsonl)
    selfcheck = load_jsonl(args.selfcheck_choices)
    triage = load_jsonl(args.triage_choices)
    full_rows = load_jsonl(args.full_preds)
    text_rows = load_jsonl(args.text_preds)
    param_rows = load_jsonl(args.param_preds)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.choices_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as fout, \
         open(args.choices_jsonl, "w", encoding="utf-8") as fchoices:

        for rid, brow in bench.items():
            s_dec = selfcheck[rid]["decision"]
            t_dec = triage[rid]["decision"]

                                        
            if s_dec == "FALLBACK":
                chosen = param_rows[rid]
                chosen_source = "parametric"
                final_dec = "FALLBACK"

                                                                       
            else:
                if t_dec == "TEXT_ONLY":
                    chosen = text_rows[rid]
                    chosen_source = "text_only"
                    final_dec = "TEXT_ONLY"
                elif t_dec == "FULL_MM":
                    chosen = full_rows[rid]
                    chosen_source = "full_mm"
                    final_dec = "FULL_MM"
                else:
                    chosen = param_rows[rid]
                    chosen_source = "parametric"
                    final_dec = "FALLBACK"

            pred_row = {
                "row_id": chosen["row_id"],
                "qid": chosen["qid"],
                "question": chosen["question"],
                "regime": chosen["regime"],
                "baseline": "cascaded_router",
                "prediction_model": model_name_from_prediction(chosen),
                "prediction": chosen["prediction"],
                "error": chosen.get("error", ""),
            }

            choice_row = {
                "row_id": rid,
                "regime": brow["regime"],
                "stage1_selfcheck": s_dec,
                "stage2_triage": t_dec,
                "decision": final_dec,
                "chosen_source": chosen_source,
            }

            fout.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
            fchoices.write(json.dumps(choice_row, ensure_ascii=False) + "\n")

    print(f"Wrote cascaded predictions -> {args.output_jsonl}")
    print(f"Wrote cascaded choices -> {args.choices_jsonl}")

if __name__ == "__main__":
    main()
