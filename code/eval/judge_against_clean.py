#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_missing_row_id(row_id) -> bool:
    return row_id is None or (isinstance(row_id, str) and not row_id.strip())


def is_auth_error(msg: str) -> bool:
    s = (msg or "").lower()
    return (
        "invalid_api_key" in s
        or "incorrect api key provided" in s
        or "authentication" in s
        or "unauthorized" in s
        or "status': 401" in s
        or "status\": 401" in s
    )


def is_retryable_error(msg: str) -> bool:
    s = (msg or "").lower()
    retry_markers = (
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "api_connection_error",
        "service unavailable",
    )
    return any(m in s for m in retry_markers)


def build_clean_reference_map(benchmark_jsonl: str) -> Dict[int, dict]:
    ref = {}
    for row in iter_jsonl(benchmark_jsonl):
        if row.get('regime') == 'TC_IC':
            ref[int(row['qid'])] = row
    return ref


def build_reference_text(row: dict) -> str:
    lines = []
    lines.append('TRUSTED CLEAN TEXT EVIDENCE:')
    for i, ev in enumerate(row.get('text_evidence', []), start=1):
        lines.append(f'[{i}] Title: {ev.get("title", "")}')
        lines.append(f'    Snippet: {ev.get("snippet", "")}')
        if ev.get('url'):
            lines.append(f'    URL: {ev.get("url", "")}')
    img = row.get('image_evidence', {})
    lines.append('')
    lines.append('TRUSTED CLEAN IMAGE EVIDENCE:')
    lines.append(f"Title: {img.get('title','')}")
    lines.append(f"Alt/caption: {img.get('alt_text','')}")
    if img.get('page_url'):
        lines.append(f"Source page: {img.get('page_url','')}")
    return '\n'.join(lines)


def judge_prompt(question: str, answer: str, ref_text: str) -> str:
    return (
        'You are evaluating factuality for a research benchmark.\n'
        'Given a question, a model answer, and trusted clean evidence, decide whether the answer is factually supported.\n'
        'Use only the trusted clean evidence below.\n'
        'Return strict JSON only with keys: label, score, reason.\n'
        'Allowed labels: supported, partially_supported, unsupported, uncertain.\n'
        'Use scores: supported=1.0, partially_supported=0.5, unsupported=0.0, uncertain=0.25.\n\n'
        f'QUESTION:\n{question}\n\n'
        f'MODEL ANSWER:\n{answer}\n\n'
        f'{ref_text}\n\n'
        'Return JSON only.'
    )


def make_client(api_base: str = '', api_key_env: str = '') -> OpenAI:
    if OpenAI is None:
        raise SystemExit(
            "Missing dependency: openai. Install requirements_mm.txt or requirements.txt before running this script."
        )
    kwargs = {}
    base_url = api_base or os.environ.get('OPENAI_BASE_URL', '')
    if base_url:
        kwargs['base_url'] = base_url
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f'{api_key_env} is not set')
        kwargs['api_key'] = api_key
    return OpenAI(**kwargs)


def parse_jsonish(text: str) -> Tuple[str, float, str]:
    text = text.strip()
    try:
        obj = json.loads(text)
        return str(obj.get('label', 'uncertain')), float(obj.get('score', 0.25)), str(obj.get('reason', ''))
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return str(obj.get('label', 'uncertain')), float(obj.get('score', 0.25)), str(obj.get('reason', ''))
        except Exception:
            pass
    return 'uncertain', 0.25, text[:300]


def call_judge(client: OpenAI, model: str, prompt: str, backend: str, timeout_s: float, max_tokens: int) -> str:
    if backend == 'responses':
        resp = client.with_options(timeout=timeout_s).responses.create(
            model=model,
            input=prompt,
        )
        return getattr(resp, 'output_text', '') or ''

    resp = client.with_options(timeout=timeout_s).chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark_jsonl', required=True)
    ap.add_argument('--predictions_jsonl', required=True)
    ap.add_argument('--output_jsonl', required=True)
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--sleep_s', type=float, default=0.0)
    ap.add_argument('--max_items', type=int, default=0)
    ap.add_argument('--baseline_name', default='')
    ap.add_argument('--request_timeout_s', type=float, default=90.0)
    ap.add_argument('--max_retries', type=int, default=3)
    ap.add_argument('--retry_backoff_s', type=float, default=2.0)
    ap.add_argument('--progress_every', type=int, default=25)
    ap.add_argument('--api_base', default='', help='Optional OpenAI-compatible API base URL')
    ap.add_argument('--api_key_env', default='', help='Optional environment variable containing the API key')
    ap.add_argument('--backend', choices=['responses', 'chat'], default='responses')
    ap.add_argument('--max_tokens', type=int, default=256)
    ap.add_argument(
        '--retry_error_rows',
        action='store_true',
        help='With --resume, re-judge rows whose existing judged records contain non-empty error',
    )
    ap.add_argument(
        '--allow_auth_errors',
        action='store_true',
        help='Continue on auth failures and write uncertain rows (default is fail-fast on auth errors)',
    )
    args = ap.parse_args()

    client = make_client(args.api_base, args.api_key_env)
    ref_map = build_clean_reference_map(args.benchmark_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        for rec in iter_jsonl(str(out_path)):
            row_id = rec.get('row_id')
            if is_missing_row_id(row_id):
                continue
            if args.retry_error_rows and rec.get('error'):
                continue
            done.add(row_id)

                                                                 
    if not args.allow_auth_errors:
        try:
            call_judge(client, args.model, 'healthcheck', args.backend, args.request_timeout_s, args.max_tokens)
        except Exception as exc:
            msg = str(exc)
            if is_auth_error(msg):
                raise SystemExit(
                    'OpenAI judge auth failed before run start. '
                    'Set a valid OPENAI_API_KEY or pass --allow_auth_errors to keep legacy behavior.'
                )

    n = 0
    with out_path.open('a', encoding='utf-8') as fout:
        for pred in iter_jsonl(args.predictions_jsonl):
            row_id = pred.get('row_id')
            if is_missing_row_id(row_id):
                continue
            if row_id in done:
                continue
            if args.max_items and n >= args.max_items:
                break
            qid = int(pred['qid'])
            ref = ref_map.get(qid)
            if ref is None:
                continue
            answer = pred.get('prediction', '') or ''
            prompt = judge_prompt(pred['question'], answer, build_reference_text(ref))
            label, score, reason = 'uncertain', 0.25, 'no_output'
            error = ''
            for attempt in range(1, max(1, args.max_retries) + 1):
                try:
                    out_text = call_judge(
                        client,
                        args.model,
                        prompt,
                        args.backend,
                        args.request_timeout_s,
                        args.max_tokens,
                    )
                    label, score, reason = parse_jsonish(out_text)
                    error = ''
                    break
                except Exception as exc:
                    error = str(exc)
                    if not args.allow_auth_errors and is_auth_error(error):
                        raise SystemExit(
                            'OpenAI judge auth failed mid-run. '
                            'Stopping to avoid writing misleading uncertain rows. '
                            'Fix OPENAI_API_KEY and rerun with --resume --retry_error_rows.'
                        )
                    if attempt >= max(1, args.max_retries) or not is_retryable_error(error):
                        break
                    time.sleep(max(0.0, args.retry_backoff_s) * attempt)
            rec = {
                'row_id': row_id,
                'qid': qid,
                'question': pred.get('question', ''),
                'regime': pred.get('regime', ''),
                'baseline': args.baseline_name or Path(args.predictions_jsonl).stem,
                'prediction_model': pred.get('model', ''),
                'judge_model': args.model,
                'label': label,
                'score': score,
                'reason': reason,
                'prediction': answer,
                'error': error,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fout.flush()
            n += 1
            if args.progress_every > 0 and n % args.progress_every == 0:
                print(f'Progress: wrote {n} new judged rows -> {out_path}', flush=True)
            if args.sleep_s:
                time.sleep(args.sleep_s)
    print(f'Wrote judged results to {out_path}')


if __name__ == '__main__':
    main()
