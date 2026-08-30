import argparse, json, copy

TEXT_KEYS = {
    'text_evidence','clean_text_evidence','polluted_text_evidence','passages','retrieved_text','texts','evidence'
}
IMAGE_KEYS = {
    'image','images','image_evidence','clean_image','polluted_image','image_path','image_local_path','local_image_path',
    'image_url','image_urls','clean_image_path','polluted_image_path','clean_image_url','polluted_image_url',
    'image_alt','image_title','image_metadata'
}
KEEP_ALWAYS = {'id','qid','question_id','dataset','regime','question','prompt','query','answer','gold_answer','reference_answer'}

def blank_value(v):
    if isinstance(v, list):
        return []
    if isinstance(v, dict):
        return {}
    return None

def transform(row, mode):
    row = copy.deepcopy(row)
    for k in list(row.keys()):
        kl = k.lower()
        if k in KEEP_ALWAYS:
            continue
        is_text = k in TEXT_KEYS or ('text' in kl and 'question' not in kl and 'answer' not in kl)
        is_image = k in IMAGE_KEYS or 'image' in kl or kl in {'alt','title','url'} and 'question' not in kl
        if mode == 'parametric':
            if is_text or is_image or 'evidence' in kl or 'retriev' in kl or 'caption' in kl:
                row[k] = blank_value(row[k])
        elif mode == 'text_only':
            if is_image or 'caption' in kl:
                row[k] = blank_value(row[k])
        elif mode == 'image_only':
            if is_text or 'passage' in kl or 'retriev' in kl:
                row[k] = blank_value(row[k])
        else:
            raise ValueError(mode)
    return row

ap = argparse.ArgumentParser()
ap.add_argument('--input_jsonl', required=True)
ap.add_argument('--output_jsonl', required=True)
ap.add_argument('--mode', required=True, choices=['parametric','text_only','image_only'])
args = ap.parse_args()

with open(args.input_jsonl, 'r', encoding='utf-8') as f, open(args.output_jsonl, 'w', encoding='utf-8') as out:
    n=0
    for line in f:
        line=line.strip()
        if not line:
            continue
        row=json.loads(line)
        out.write(json.dumps(transform(row, args.mode), ensure_ascii=False)+'\n')
        n+=1
print('wrote', n, 'rows to', args.output_jsonl)
