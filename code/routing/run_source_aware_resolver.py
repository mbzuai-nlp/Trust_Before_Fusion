import argparse
import json
import os
import re
import time
from pathlib import Path
from openai import OpenAI

def load_jsonl(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows[obj["row_id"]] = obj
    return rows

def parse_json(text):
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass

    return None

def normalize_decision(x):
    x = str(x or "").upper().strip()
    if "FULL" in x:
        return "FULL_MM"
    if "TEXT" in x:
        return "TEXT_ONLY"
    if "PARAM" in x or "FALLBACK" in x:
        return "FALLBACK"
    if "COMPOSE" in x or "SYNTH" in x:
        return "COMPOSE"
    return "FALLBACK"

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

def evidence_block(row):
    parts = []

    parts.append("TEXT EVIDENCE:\n")
    for i, t in enumerate(row.get("text_evidence", [])[:5], start=1):
        parts.append(f"[T{i}] title: {t.get('title','')}\n")
        parts.append(f"[T{i}] snippet: {t.get('snippet','')}\n")
        if t.get("url"):
            parts.append(f"[T{i}] url: {t.get('url','')}\n")
        parts.append("\n")

    img = row.get("image_evidence", {}) or {}
    parts.append("IMAGE EVIDENCE METADATA:\n")
    parts.append(f"[I1] title: {img.get('title','')}\n")
    parts.append(f"[I1] alt_text: {img.get('alt_text','')}\n")
    parts.append(f"[I1] page_url: {img.get('page_url','')}\n")
    return "".join(parts)

STRUCTURED_FIELDS = {
    "text_reliability",
    "image_reliability",
    "internal_external_conflict",
    "cross_modal_conflict",
    "candidate_scores",
}


def make_prompt(row, param_ans, text_ans, full_ans, compose_allowed, disabled_fields=None):
    disabled_fields = set(disabled_fields or [])
    compose_rule = (
        "You may choose COMPOSE if no single candidate is fully safe. If you choose COMPOSE, write a new 3-6 sentence answer using only claims supported by reliable sources."
        if compose_allowed
        else "You must choose one of FULL_MM, TEXT_ONLY, or FALLBACK. Do not choose COMPOSE."
    )
    definitions = []
    if "text_reliability" not in disabled_fields:
        definitions.append("- text_reliability: trustworthy | conflicting | weak | suspicious")
    if "image_reliability" not in disabled_fields:
        definitions.append("- image_reliability: trustworthy | suspicious | weak | irrelevant")
    if "internal_external_conflict" not in disabled_fields:
        definitions.append("- internal_external_conflict: yes | no | unclear")
    if "cross_modal_conflict" not in disabled_fields:
        definitions.append("- cross_modal_conflict: yes | no | unclear")

    ablation_note = ""
    if disabled_fields:
        ablation_note = (
            "\nAblation setting: do not output or explicitly use these structured criteria: "
            + ", ".join(sorted(disabled_fields))
            + ". Make the decision using only the remaining requested criteria.\n"
        )

    json_fields = []
    if "text_reliability" not in disabled_fields:
        json_fields.append('  "text_reliability": "...",')
    if "image_reliability" not in disabled_fields:
        json_fields.append('  "image_reliability": "...",')
    if "internal_external_conflict" not in disabled_fields:
        json_fields.append('  "internal_external_conflict": "...",')
    if "cross_modal_conflict" not in disabled_fields:
        json_fields.append('  "cross_modal_conflict": "...",')
    if "candidate_scores" not in disabled_fields:
        json_fields.extend([
            '  "candidate_scores": {',
            '    "FALLBACK": 0,',
            '    "TEXT_ONLY": 0,',
            '    "FULL_MM": 0',
            '  },',
        ])
    json_fields.extend([
        '  "decision": "FULL_MM|TEXT_ONLY|FALLBACK|COMPOSE",',
        '  "chosen_source": "full_mm|text_only|parametric|composed",',
        '  "final_answer": "...",',
        '  "reason": "..."',
    ])
    json_schema = "{\n" + "\n".join(json_fields) + "\n}"

    return f"""
You are a source-aware multimodal RAG conflict resolver.

Your job is to decide how to answer a long-form question when retrieved evidence may be unreliable.
Do not trust evidence just because it is topically relevant.
Do not use hidden benchmark metadata. Judge only the content shown below.

QUESTION:
{row.get("question","")}

{evidence_block(row)}

CANDIDATE ANSWERS:

[FALLBACK / PARAMETRIC]
{param_ans}

[TEXT_ONLY]
{text_ans}

[FULL_MM]
{full_ans}

TASK:
Analyze whether the text evidence, image evidence, and candidate answers are mutually reliable.

Use these definitions:
{chr(10).join(definitions)}
{ablation_note}

Decision choices:
- FULL_MM: choose the full multimodal answer
- TEXT_ONLY: choose the text-only answer
- FALLBACK: choose the parametric answer
- COMPOSE: write a corrected answer using only reliable claims

Rules:
1. Prefer FULL_MM only when text and image evidence are both useful and mutually consistent.
2. Prefer TEXT_ONLY when text is useful but image evidence is suspicious, weak, or irrelevant.
3. Prefer FALLBACK when retrieved text itself appears unreliable, conflicting, or deceptive.
4. Prefer COMPOSE when candidates contain complementary supported parts but no candidate is fully reliable.
5. {compose_rule}
6. The final answer must not mention internal benchmark metadata.

Return JSON only:
{json_schema}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_jsonl", required=True)
    ap.add_argument("--param_preds", required=True)
    ap.add_argument("--text_preds", required=True)
    ap.add_argument("--full_preds", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--choices_jsonl", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--mode", choices=["compose", "select_only"], default="compose")
    ap.add_argument("--sleep_s", type=float, default=0.3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--api_base", default="")
    ap.add_argument("--api_key_env", default="")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--request_timeout_s", type=float, default=90.0)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--retry_backoff_s", type=float, default=2.0)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument(
        "--disable_fields",
        nargs="*",
        choices=sorted(STRUCTURED_FIELDS),
        default=[],
        help="Structured resolver fields to remove from the prompt/output for SATR ablations.",
    )
    args = ap.parse_args()

    client = make_client(args.api_base, args.api_key_env)

    bench = load_jsonl(args.benchmark_jsonl)
    param = load_jsonl(args.param_preds)
    text = load_jsonl(args.text_preds)
    full = load_jsonl(args.full_preds)

    done = set()
    if args.resume and Path(args.choices_jsonl).exists():
        with open(args.choices_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["row_id"])
                except Exception:
                    pass

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.choices_jsonl).parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"
    written = 0

    with open(args.output_jsonl, mode, encoding="utf-8") as fout, open(args.choices_jsonl, mode, encoding="utf-8") as fchoices:
        for rid, row in bench.items():
            if rid in done:
                continue
            if rid not in param or rid not in text or rid not in full:
                continue

            p_ans = param[rid].get("prediction", "")
            t_ans = text[rid].get("prediction", "")
            f_ans = full[rid].get("prediction", "")

            prompt = make_prompt(
                row,
                p_ans,
                t_ans,
                f_ans,
                compose_allowed=(args.mode == "compose"),
                disabled_fields=args.disable_fields,
            )

            raw = ""
            obj = None
            last_error = ""
            for attempt in range(1, max(1, args.max_retries) + 1):
                try:
                    resp = client.with_options(timeout=args.request_timeout_s).chat.completions.create(
                        model=args.model,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": "You are a careful source-aware multimodal RAG conflict resolver."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=args.max_tokens,
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                    obj = parse_json(raw)
                    last_error = ""
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt < max(1, args.max_retries):
                        time.sleep(args.retry_backoff_s * attempt)

            if last_error:
                raw = f"ERROR: {last_error}"
                obj = None

            if not isinstance(obj, dict):
                obj = {
                    "decision": "FALLBACK",
                    "chosen_source": "parametric",
                    "final_answer": p_ans,
                    "reason": "parse_failure_fallback",
                }

            decision = normalize_decision(obj.get("decision"))
            chosen_source = str(obj.get("chosen_source", "")).lower().strip()

            if args.mode == "select_only" and decision == "COMPOSE":
                decision = "FALLBACK"
                chosen_source = "parametric"

            if decision == "FULL_MM":
                final_answer = f_ans
                chosen_source = "full_mm"
            elif decision == "TEXT_ONLY":
                final_answer = t_ans
                chosen_source = "text_only"
            elif decision == "COMPOSE" and args.mode == "compose":
                final_answer = str(obj.get("final_answer", "")).strip()
                chosen_source = "composed"
                if not final_answer:
                    final_answer = p_ans
                    chosen_source = "parametric"
                    decision = "FALLBACK"
            else:
                final_answer = p_ans
                chosen_source = "parametric"
                decision = "FALLBACK"

            pred_row = {
                "row_id": rid,
                "qid": row.get("qid"),
                "question": row.get("question"),
                "regime": row.get("regime"),
                "baseline": "source_aware_conductor" if args.mode == "compose" else "source_aware_selector",
                "prediction_model": args.model,
                "prediction": final_answer,
                "error": "",
            }

            choice_row = {
                "row_id": rid,
                "qid": row.get("qid"),
                "regime": row.get("regime"),
                "decision": decision,
                "chosen_source": chosen_source,
                "text_reliability": "" if "text_reliability" in args.disable_fields else obj.get("text_reliability", ""),
                "image_reliability": "" if "image_reliability" in args.disable_fields else obj.get("image_reliability", ""),
                "internal_external_conflict": "" if "internal_external_conflict" in args.disable_fields else obj.get("internal_external_conflict", ""),
                "cross_modal_conflict": "" if "cross_modal_conflict" in args.disable_fields else obj.get("cross_modal_conflict", ""),
                "candidate_scores": {} if "candidate_scores" in args.disable_fields else obj.get("candidate_scores", {}),
                "disabled_fields": args.disable_fields,
                "reason": obj.get("reason", ""),
                "raw_output": raw,
            }

            fout.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
            fchoices.write(json.dumps(choice_row, ensure_ascii=False) + "\n")
            fout.flush()
            fchoices.flush()

            written += 1
            if args.progress_every and written % args.progress_every == 0:
                print(f"Progress: wrote {written} resolver rows -> {args.output_jsonl}", flush=True)
            if args.max_rows and written >= args.max_rows:
                break
            time.sleep(args.sleep_s)

    print("Wrote predictions ->", args.output_jsonl)
    print("Wrote choices ->", args.choices_jsonl)

if __name__ == "__main__":
    main()
