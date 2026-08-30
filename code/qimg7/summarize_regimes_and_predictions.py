import argparse, json, collections, statistics, os

def load_jsonl(path):
    out=[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out

ap=argparse.ArgumentParser()
ap.add_argument('--benchmark_jsonl', required=True)
ap.add_argument('--predictions_jsonl', default=None)
args=ap.parse_args()

bench=load_jsonl(args.benchmark_jsonl)
reg=collections.Counter()
qlens=[]
for x in bench:
    reg[x.get('regime','UNKNOWN')] += 1
    qlens.append(len(x.get('question','')))
print('benchmark_rows=', len(bench))
print('regime_counts=', dict(reg))
if qlens:
    print('avg_question_chars=', round(statistics.mean(qlens),1))

if args.predictions_jsonl and os.path.exists(args.predictions_jsonl):
    preds=load_jsonl(args.predictions_jsonl)
    preg=collections.Counter()
    lens=[]
    for x in preds:
        preg[x.get('regime','UNKNOWN')] += 1
        txt=x.get('prediction','') or x.get('response','') or x.get('output_text','') or ''
        lens.append(len(txt))
    print('prediction_rows=', len(preds))
    print('prediction_regime_counts=', dict(preg))
    if lens:
        print('avg_prediction_chars=', round(statistics.mean(lens),1))
