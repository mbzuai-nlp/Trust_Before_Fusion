#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def iter_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_missing_row_id(row_id) -> bool:
    return row_id is None or (isinstance(row_id, str) and not row_id.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--judged_jsonls', nargs='+', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--output_md', required=True)
    ap.add_argument(
        '--include_error_rows',
        action='store_true',
        help='Include judged rows with non-empty `error` in metric aggregates',
    )
    args = ap.parse_args()

                                                                 
                                                                                       
    dedup = {}
    duplicate_rows = 0
    for path in args.judged_jsonls:
        for i, row in enumerate(iter_jsonl(path), start=1):
            baseline = row.get('baseline') or Path(path).stem
            row_id = row.get('row_id')
            if is_missing_row_id(row_id):
                row_id = f'__no_row_id__{Path(path).name}__{i}'
            dkey = (baseline, row_id)
            if dkey in dedup:
                duplicate_rows += 1
            row2 = dict(row)
            row2['_baseline_norm'] = baseline
            dedup[dkey] = row2

    groups_all = defaultdict(list)
    groups_eval = defaultdict(list)
    for row in dedup.values():
        baseline = row.get('_baseline_norm') or row.get('baseline') or 'UNKNOWN'
        regime = row.get('regime', 'UNKNOWN')
        key = (baseline, regime)
        groups_all[key].append(row)
        if args.include_error_rows or not row.get('error'):
            groups_eval[key].append(row)

    rows = []
    for key in sorted(groups_all.keys()):
        baseline, regime = key
        items = groups_eval.get(key, [])
        total_items = groups_all.get(key, [])
        dropped = len(total_items) - len(items)
        scores = [float(x.get('score', 0.0)) for x in items]
        labels = [x.get('label', '') for x in items]
        n = len(items)
        if n == 0:
            mean_score = ''
            supported_rate = ''
            partial_rate = ''
            unsupported_rate = ''
            uncertain_rate = ''
        else:
            mean_score = round(mean(scores), 4)
            supported_rate = round(sum(l == 'supported' for l in labels) / n, 4)
            partial_rate = round(sum(l == 'partially_supported' for l in labels) / n, 4)
            unsupported_rate = round(sum(l == 'unsupported' for l in labels) / n, 4)
            uncertain_rate = round(sum(l == 'uncertain' for l in labels) / n, 4)
        rows.append({
            'baseline': baseline,
            'regime': regime,
            'n': n,
            'n_total': len(total_items),
            'dropped_error_rows': dropped,
            'mean_score': mean_score,
            'supported_rate': supported_rate,
            'partial_rate': partial_rate,
            'unsupported_rate': unsupported_rate,
            'uncertain_rate': uncertain_rate,
        })

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['baseline','regime','n','mean_score'])
        writer.writeheader()
        writer.writerows(rows)

    out_md = Path(args.output_md)
    with out_md.open('w', encoding='utf-8') as f:
        if rows:
            hdr = list(rows[0].keys())
            f.write('| ' + ' | '.join(hdr) + ' |\n')
            f.write('| ' + ' | '.join(['---'] * len(hdr)) + ' |\n')
            for r in rows:
                f.write('| ' + ' | '.join(str(r[h]) for h in hdr) + ' |\n')
    print(
        f'Wrote {len(rows)} summary rows -> {out_csv} and {out_md} '
        f'(deduped {duplicate_rows} duplicate judged rows)'
    )


if __name__ == '__main__':
    main()
