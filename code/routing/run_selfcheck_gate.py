import json
import argparse
import os
from pathlib import Path
from openai import OpenAI

def load_jsonl(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows[obj["row_id"]] = obj
    return rows

def make_gate_prompt(bench_row):
    q = bench_row["question"]
    texts = bench_row["text_evidence"]
    img = bench_row["image_evidence"]

    parts = []
    parts.append(f"Question: {q}\n")
    parts.append("Retrieved text evidence:\n")
    for i, t in enumerate(texts[:5], start=1):
        parts.append(f"[T{i}] title={t.get('title','')}\n")
        parts.append(f"[T{i}] snippet={t.get('snippet','')}\n")
        if t.get("url"):
            parts.append(f"[T{i}] url={t.get('url','')}\n")

    parts.append("\nRetrieved image metadata:\n")
    parts.append(f"title={img.get('title','')}\n")
    parts.append(f"alt_text={img.get('alt_text','')}\n")
    if img.get("page_url"):
        parts.append(f"page_url={img.get('page_url','')}\n")

    parts.append(
        "\nTask: Decide whether the retrieved evidence looks mutually consistent and trustworthy enough "
        "to rely on for answering the question. Ignore your own world knowledge. Judge only the evidence. "
        "Answer with JSON only: {\"decision\":\"TRUST\" or \"FALLBACK\", \"reason\":\"...\"}"
    )
    return "".join(parts)

def parse_decision(text):
    text = text.strip()
    if "FALLBACK" in text.upper():
        return "FALLBACK"
    return "TRUST"

def make_client(api_base="", api_key_env=""):
    kwargs = {}
    base_url = api_base or os.environ.get("OPENAI_BASE_URL", "")
    if base_url:
        kwargs["base_url"] = base_url
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f"{api_key_env} is not set")
        kwargs["api_key"] = api_key
    return OpenAI(**kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_jsonl", required=True)
    ap.add_argument("--full_judged", required=True)
    ap.add_argument("--param_judged", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--choices_jsonl", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--api_base", default="")
    ap.add_argument("--api_key_env", default="")
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    client = make_client(args.api_base, args.api_key_env)

    bench = load_jsonl(args.benchmark_jsonl)
    full_rows = load_jsonl(args.full_judged)
    param_rows = load_jsonl(args.param_judged)

    done = set()
    if args.resume and Path(args.choices_jsonl).exists():
        with open(args.choices_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done.add(obj["row_id"])

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.choices_jsonl).parent.mkdir(parents=True, exist_ok=True)

    out_mode = "a" if args.resume else "w"
    with open(args.output_jsonl, out_mode, encoding="utf-8") as fout, \
         open(args.choices_jsonl, out_mode, encoding="utf-8") as fchoices:

        for rid, brow in bench.items():
            if rid in done:
                continue

            prompt = make_gate_prompt(brow)
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are a retrieval trust gate."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            text = resp.choices[0].message.content
            decision = parse_decision(text)

            chosen = full_rows[rid] if decision == "TRUST" else param_rows[rid]

            pred_row = {
                "row_id": chosen["row_id"],
                "qid": chosen["qid"],
                "question": chosen["question"],
                "regime": chosen["regime"],
                "baseline": "selfcheck_gate",
                "prediction_model": chosen["prediction_model"],
                "prediction": chosen["prediction"],
                "error": "",
            }
            choice_row = {
                "row_id": rid,
                "decision": decision,
                "reasoning_raw": text,
                "chosen_source": chosen["baseline"],
            }

            fout.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
            fchoices.write(json.dumps(choice_row, ensure_ascii=False) + "\n")

    print(f"Wrote gate predictions -> {args.output_jsonl}")
    print(f"Wrote gate decisions -> {args.choices_jsonl}")

if __name__ == "__main__":
    main()
