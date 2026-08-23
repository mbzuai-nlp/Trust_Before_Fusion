#!/usr/bin/env python3
"""Run a quick multimodal baseline over regime JSONL using OpenAI Responses API.

This script is intentionally standalone so you can start experimenting immediately,
even before fully wiring MM-MIRAGE into the old repo.

It uses both:
- text evidence (top-k text snippets)
- image evidence metadata (title, alt_text, rationale)
- optional image pixels (local file or image URL)

If your regime is caption_flip, the same image pixels are paired with false metadata.
If your regime is typographic_attack, the image pixels themselves are polluted.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import Iterable, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


IMAGE_INPUT_ERROR_MARKERS = (
    'does not represent a valid image',
    'error while downloading',
    "'param': 'url'",
    '"param": "url"',
    "'param': 'input'",
    '"param": "input"',
    'unsupported image',
    'expected a valid url',
    'invalid format',
)


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_data_url(local_path: str) -> str:
    mime, _ = mimetypes.guess_type(local_path)
    mime = mime or 'image/jpeg'
    raw = Path(local_path).read_bytes()
    b64 = base64.b64encode(raw).decode('utf-8')
    return f'data:{mime};base64,{b64}'


def detect_image_mime(raw: bytes, fallback_ref: str = '', header_content_type: str = '') -> str:
    ctype = header_content_type.split(';', 1)[0].strip().lower()
    if ctype in {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}:
        return ctype
    if raw.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if raw.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    if raw.startswith(b'RIFF') and raw[8:12] == b'WEBP':
        return 'image/webp'
    return ''


def url_to_data_url(url: str, timeout_s: float = 20.0, max_bytes: int = 20 * 1024 * 1024) -> str:
    req = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (compatible; MM-MIRAGE/1.0)',
            'Accept': 'image/avif,image/webp,image/png,image/jpeg,image/gif,*/*;q=0.8',
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(f'image too large: {url}')
        mime = detect_image_mime(raw, fallback_ref=url, header_content_type=resp.headers.get('Content-Type', ''))
        if not mime:
            raise ValueError(f'unsupported image response: {url}')
    b64 = base64.b64encode(raw).decode('utf-8')
    return f'data:{mime};base64,{b64}'


def resolve_local_image_path(img: dict) -> str:
    """Return the first existing local image path from supported fields."""
    candidates = (
        img.get('image_local_path', ''),
        img.get('resolved_local_path', ''),
        img.get('local_path', ''),
    )
    for candidate in candidates:
        path = str(candidate or '').strip()
        if path and Path(path).exists():
            return path
    return ''


def resolve_remote_image_url(img: dict) -> str:
    for key in ('image_url', 'url'):
        ref = str(img.get(key, '') or '').strip()
        if is_supported_image_ref(ref):
            return ref
    return ''


def is_supported_image_ref(ref: str) -> bool:
    if not ref:
        return False
    if ref.startswith('data:image/'):
        return True
    parsed = urlparse(ref)
    return parsed.scheme in {'http', 'https'} and bool(parsed.netloc)


def resolve_image_reference(
    img: dict,
    fetch_remote: bool = False,
    allow_remote_url: bool = True,
) -> str:
    """Prefer local pixels; fall back to URL fields when no local file exists."""
    local_path = resolve_local_image_path(img)
    if local_path:
        return to_data_url(local_path)
    ref = resolve_remote_image_url(img)
    if ref and fetch_remote:
        return url_to_data_url(ref)
    if ref and allow_remote_url:
        return ref
    return ''


def is_image_input_error(message: str) -> bool:
    msg = (message or '').lower()
    return any(marker in msg for marker in IMAGE_INPUT_ERROR_MARKERS)


def build_request_content(
    row: dict,
    include_pixels: bool,
    fetch_remote: bool = False,
    allow_remote_url: bool = True,
) -> tuple[List[dict], bool, str]:
    content: List[dict] = []
    content.append({'type': 'input_text', 'text': build_text_prompt(row)})
    if not include_pixels:
        return content, False, ''

    img = row.get('image_evidence', {})
    try:
        image_ref = resolve_image_reference(
            img,
            fetch_remote=fetch_remote,
            allow_remote_url=allow_remote_url,
        )
    except Exception:
        return content, False, ''
    if image_ref:
        content.append({'type': 'input_image', 'image_url': image_ref, 'detail': 'auto'})
        if image_ref.startswith('data:'):
            source = 'url_data' if fetch_remote and not resolve_local_image_path(img) else 'local'
        else:
            source = 'url'
        return content, True, source
    return content, False, ''


def build_chat_content(responses_content: List[dict]) -> List[dict]:
    """Convert Responses API content to OpenAI-compatible Chat Completions content."""
    chat_content: List[dict] = []
    for item in responses_content:
        if item.get('type') == 'input_text':
            chat_content.append({'type': 'text', 'text': item.get('text', '')})
        elif item.get('type') == 'input_image':
            chat_content.append({
                'type': 'image_url',
                'image_url': {'url': item.get('image_url', '')},
            })
    return chat_content


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


def call_model(client: OpenAI, model: str, content: List[dict], backend: str, max_tokens: int = 0) -> str:
    if backend == 'responses':
        resp = client.responses.create(
            model=model,
            input=[{'role': 'user', 'content': content}],
        )
        return getattr(resp, 'output_text', '') or ''

    kwargs = {}
    if max_tokens > 0:
        kwargs['max_tokens'] = max_tokens
    resp = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': build_chat_content(content)}],
        temperature=0.0,
        **kwargs,
    )
    return resp.choices[0].message.content or ''


def build_text_prompt(row: dict) -> str:
    lines = []
    lines.append('Answer the question using the provided evidence.')
    lines.append('Prefer information that is consistent across sources. If the evidence is unreliable or conflicting, say you are uncertain.')
    lines.append('Do not discuss benchmark construction metadata.')
    lines.append('')
    lines.append(f"QUESTION: {row['question']}")
    lines.append('')
    lines.append('TEXT EVIDENCE:')
    for i, ev in enumerate(row.get('text_evidence', []), start=1):
        lines.append(f"[{i}] Title: {ev.get('title','')}")
        lines.append(f"    Snippet: {ev.get('snippet','')}")
        lines.append(f"    URL: {ev.get('url','')}")
    img = row.get('image_evidence', {})
    lines.append('')
    lines.append('IMAGE EVIDENCE METADATA:')
    lines.append(f"Title: {img.get('title','')}")
    lines.append(f"Alt text / caption: {img.get('alt_text','')}")
    if img.get('page_url'):
        lines.append(f"Source page: {img.get('page_url','')}")
    lines.append('')
    lines.append('Now answer in 3-6 sentences, focusing on factual accuracy.')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark_jsonl', required=True)
    ap.add_argument('--output_jsonl', required=True)
    ap.add_argument('--model', required=True, help='Any vision-capable model available in your account')
    ap.add_argument('--max_items', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--sleep_s', type=float, default=0.0)
    ap.add_argument('--max_retries', type=int, default=3)
    ap.add_argument('--no_pixels', action='store_true', help='Only send text/metadata, no image pixels')
    ap.add_argument('--no_remote_images', action='store_true', help='Do not send remote image URLs when no local image file exists')
    ap.add_argument('--fetch_remote_images', action='store_true', help='Fetch remote image URLs client-side and send them as data URLs')
    ap.add_argument(
        '--backend',
        choices=['responses', 'chat'],
        default='responses',
        help='Use responses for OpenAI native models; use chat for OpenAI-compatible hosted models.',
    )
    ap.add_argument('--api_base', default='', help='Optional OpenAI-compatible API base URL')
    ap.add_argument('--api_key_env', default='', help='Optional environment variable containing the API key')
    ap.add_argument('--max_tokens', type=int, default=512, help='Chat backend max_tokens; ignored when <=0')
    args = ap.parse_args()

    client = make_client(args.api_base, args.api_key_env)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if args.resume and out_path.exists():
        for rec in iter_jsonl(str(out_path)):
            done_ids.add(rec['row_id'])

    n = 0
    with out_path.open('a', encoding='utf-8') as fout:
        for row in iter_jsonl(args.benchmark_jsonl):
            if args.max_items and n >= args.max_items:
                break
            if row['row_id'] in done_ids:
                continue

            pred = ''
            err = ''
            used_pixels = False
            image_ref_source = ''
            image_fallback_error = ''
            for attempt in range(1, args.max_retries + 1):
                content, used_pixels, image_ref_source = build_request_content(
                    row,
                    include_pixels=(not args.no_pixels),
                    fetch_remote=args.fetch_remote_images,
                    allow_remote_url=(not args.no_remote_images),
                )
                try:
                    pred = call_model(client, args.model, content, args.backend, args.max_tokens)
                    break
                except Exception as exc:
                    err = str(exc)
                    if used_pixels and is_image_input_error(err):
                        image_fallback_error = err
                        if image_ref_source == 'url':
                            content, used_pixels, image_ref_source = build_request_content(
                                row,
                                include_pixels=True,
                                fetch_remote=True,
                                allow_remote_url=(not args.no_remote_images),
                            )
                            if used_pixels:
                                try:
                                    pred = call_model(client, args.model, content, args.backend, args.max_tokens)
                                    err = ''
                                    break
                                except Exception as fetched_exc:
                                    err = str(fetched_exc)
                        try:
                            content, used_pixels, image_ref_source = build_request_content(row, include_pixels=False)
                            pred = call_model(client, args.model, content, args.backend, args.max_tokens)
                            err = ''
                            break
                        except Exception as fallback_exc:
                            err = str(fallback_exc)
                    if attempt == args.max_retries:
                        break
                    time.sleep(min(8, attempt * 2))

            record = {
                'row_id': row['row_id'],
                'qid': row['qid'],
                'question': row['question'],
                'regime': row['regime'],
                'model': args.model,
                'used_pixels': used_pixels,
                'image_ref_source': image_ref_source,
                'image_fallback_error': image_fallback_error,
                'prediction': pred,
                'error': err,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            fout.flush()
            n += 1
            if args.sleep_s:
                time.sleep(args.sleep_s)

    print(f'Wrote predictions to {out_path}')


if __name__ == '__main__':
    main()
