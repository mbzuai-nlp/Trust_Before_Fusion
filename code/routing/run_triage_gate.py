import json
import argparse
import os
import re
from pathlib import Path
from openai import OpenAI

ALLOWED_DECISIONS = {"FULL_MM", "TEXT_ONLY", "FALLBACK"}

def load_jsonl(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows[obj["row_id"]] = obj
    return rows

def model_facing_evidence(bench_row):
    """Return only fields available to a normal retrieval-time router."""
    texts = []
    for text in bench_row.get("text_evidence", [])[:5]:
        texts.append({
            "title": str(text.get("title", "") or ""),
            "snippet": str(text.get("snippet", "") or ""),
            "url": str(text.get("url", "") or ""),
        })

    image = bench_row.get("image_evidence", {}) or {}
    return {
        "question": str(bench_row.get("question", "") or ""),
        "text_evidence": texts,
        "image_evidence": {
            "title": str(image.get("title", "") or ""),
            "alt_text": str(image.get("alt_text", "") or ""),
            "page_url": str(image.get("page_url", "") or ""),
        },
    }


def make_prompt(bench_row):
    evidence = model_facing_evidence(bench_row)
    q = evidence["question"]
    texts = evidence["text_evidence"]
    img = evidence["image_evidence"]

    lines = []
    lines.append(f"Question: {q}\n")
    lines.append("Retrieved text evidence:\n")
    for i, t in enumerate(texts[:5], start=1):
        lines.append(f"[T{i}] title={t.get('title','')}\n")
        lines.append(f"[T{i}] snippet={t.get('snippet','')}\n")
        if t.get("url"):
            lines.append(f"[T{i}] url={t.get('url','')}\n")

    lines.append("\nRetrieved image evidence metadata:\n")
    lines.append(f"title={img.get('title','')}\n")
    lines.append(f"alt_text={img.get('alt_text','')}\n")
    lines.append(f"source_page={img.get('page_url','')}\n")

    lines.append(
        "\nDecide how the system should answer.\n"
        "Choose exactly one:\n"
        "- FULL_MM: use both text and image evidence\n"
        "- TEXT_ONLY: ignore image evidence, rely on text evidence only\n"
        "- FALLBACK: evidence is unreliable, answer parametrically\n\n"
        "Rules:\n"
        "1. Choose FULL_MM only if text and image evidence both look relevant and mutually trustworthy.\n"
        "2. Choose TEXT_ONLY if the text looks useful but the image evidence looks suspicious, weak, irrelevant, or possibly manipulated.\n"
        "3. Choose FALLBACK if the text evidence itself looks mutually inconsistent, deceptive, or unreliable.\n"
        "4. Ignore your own world knowledge; judge only the evidence quality.\n"
        "Return JSON only: {\"decision\":\"FULL_MM\"|\"TEXT_ONLY\"|\"FALLBACK\",\"reason\":\"...\"}"
    )
    return "".join(lines)


def build_request_payload(bench_row, model="gpt-4o-mini", max_tokens=256):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a retrieval trust gate."},
            {"role": "user", "content": make_prompt(bench_row)},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

def parse_decision(text):
    try:
        obj = json.loads(text.strip())
        decision = str(obj.get("decision", "")).upper().strip()
        if decision in ALLOWED_DECISIONS:
            return decision
    except Exception:
        pass
    match = re.search(r'"decision"\s*:\s*"(FULL_MM|TEXT_ONLY|FALLBACK)"', text.upper())
    return match.group(1) if match else "FALLBACK"

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
    ap.add_argument("--full_preds", required=True)
    ap.add_argument("--text_preds", required=True)
    ap.add_argument("--param_preds", required=True)
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
    full_rows = load_jsonl(args.full_preds)
    text_rows = load_jsonl(args.text_preds)
    param_rows = load_jsonl(args.param_preds)

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

            request = build_request_payload(brow, args.model, args.max_tokens)
            resp = client.chat.completions.create(**request)
            raw = resp.choices[0].message.content
            decision = parse_decision(raw)

            if decision == "TEXT_ONLY":
                chosen = text_rows[rid]
                chosen_source = "text_only"
            elif decision == "FALLBACK":
                chosen = param_rows[rid]
                chosen_source = "parametric"
            else:
                chosen = full_rows[rid]
                chosen_source = "full_mm"

            pred_row = {
                "row_id": chosen["row_id"],
                "qid": chosen["qid"],
                "question": chosen["question"],
                "regime": chosen["regime"],
                "baseline": "triage_gate",
                "prediction_model": chosen.get("prediction_model", chosen.get("model", args.model)),
                "prediction": chosen["prediction"],
                "error": "",
            }

            choice_row = {
                "row_id": rid,
                "decision": decision,
                "chosen_source": chosen_source,
                "reasoning_raw": raw,
            }

            fout.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
            fchoices.write(json.dumps(choice_row, ensure_ascii=False) + "\n")

    print(f"Wrote triage predictions -> {args.output_jsonl}")
    print(f"Wrote triage decisions -> {args.choices_jsonl}")

if __name__ == "__main__":
    main()
