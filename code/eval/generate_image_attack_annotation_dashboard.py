#!/usr/bin/env python3
"""Generate the QIMG-7 image-attack annotation interface for the frozen sample."""

import argparse
import csv
import json
import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = REPO_ROOT / "evaluation/human_validation/image_attack_iaa/annotator_1.csv"
ROW_ID = re.compile(r"^(?P<dataset>.+)_q(?P<question>\d+)_i(?P<image>\d+)_(?P<attack>.+)$")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_items(sample_path, repo_root, output_path):
    samples = read_csv(sample_path)
    pools = {}
    items = []
    for sample in samples:
        match = ROW_ID.match(sample["row_id"])
        if not match:
            raise SystemExit(f"Invalid row_id: {sample['row_id']}")
        dataset = match.group("dataset")
        if dataset not in pools:
            pool_path = repo_root / "benchmark" / "pool" / f"{dataset}_query_image_polluted.csv"
            pools[dataset] = {
                (int(row["question_idx"]), int(row["image_idx"]), row["pollution_type"]): row
                for row in read_csv(pool_path)
            }
        key = (int(match.group("question")), int(match.group("image")), match.group("attack"))
        row = pools[dataset].get(key)
        if row is None:
            raise SystemExit(f"Sample is missing from the released pool: {sample['row_id']}")

        local_path = row.get("polluted_image_path", "").strip()
        if local_path:
            local_path = Path(os.path.relpath(repo_root / local_path, output_path.parent)).as_posix()
        items.append({
            "row_id": sample["row_id"],
            "dataset": dataset,
            "pollution_type": row["pollution_type"],
            "question": row["question"],
            "original_image_url": row.get("original_image_url", ""),
            "original_alt": row.get("original_alt", ""),
            "polluted_image_url": row.get("polluted_image_url", ""),
            "polluted_alt": row.get("polluted_alt", ""),
            "polluted_image_path": local_path,
            "manipulation_method": row.get("manipulation_method", ""),
            "manipulation_prompt": row.get("manipulation_prompt", ""),
            "on_topic": "",
            "fact_flipped": "",
            "plausible": "",
            "notes": "",
        })
    if len(items) != 56:
        raise SystemExit(f"Expected 56 frozen annotation items, found {len(items)}")
    return items


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QIMG-7 Image-Attack Annotation</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;background:#f5f6f8;color:#202124}header{position:sticky;top:0;background:#202124;color:#fff;padding:12px 20px;display:flex;gap:16px;align-items:center;z-index:2}header span{flex:1}button{padding:8px 13px;border:1px solid #bbb;border-radius:7px;background:#fff;cursor:pointer}.export{background:#16883f;color:#fff;border:0}main{max-width:1100px;margin:20px auto;padding:0 16px}.meta,.panel{background:#fff;border:1px solid #ddd;border-radius:10px;padding:14px;margin-bottom:14px}.badges{display:flex;gap:8px;margin-bottom:8px}.badge{background:#e8eefb;border-radius:12px;padding:3px 9px;font-size:12px;font-weight:600}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}.image-card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px}.image-box{height:330px;background:#eee;display:flex;align-items:center;justify-content:center}.image-box img{max-width:100%;max-height:100%;object-fit:contain}.caption{font-size:13px;color:#555;margin-top:8px;white-space:pre-wrap}.field{margin:16px 0}.field strong{display:block;margin-bottom:7px}.choices{display:flex;gap:8px}.choice.selected{background:#dce9ff;border-color:#3975d1}.nav{display:flex;gap:8px;align-items:center;margin-top:14px}textarea{box-sizing:border-box;width:100%;min-height:70px;padding:8px}@media(max-width:700px){.images{grid-template-columns:1fr}.image-box{height:250px}}
</style>
</head>
<body>
<header><strong>QIMG-7 Image-Attack Annotation</strong><span id="progress"></span><button class="export" onclick="exportCsv()">Export CSV</button></header>
<main>
  <section class="meta">
    <div class="badges"><span class="badge" id="dataset"></span><span class="badge" id="attack"></span></div>
    <h2 id="question"></h2>
    <div><strong>Method:</strong> <span id="method"></span></div>
    <div class="caption" id="prompt"></div>
  </section>
  <section class="images">
    <div class="image-card"><strong>Original</strong><div class="image-box"><img id="original-image"></div><div class="caption" id="original-caption"></div></div>
    <div class="image-card"><strong>Polluted</strong><div class="image-box"><img id="polluted-image"></div><div class="caption" id="polluted-caption"></div></div>
  </section>
  <section class="panel">
    <div class="field" data-field="on_topic"><strong>Is the polluted image on-topic for the question?</strong><div class="choices"></div></div>
    <div class="field" data-field="fact_flipped"><strong>Does the image convey a false or flipped fact?</strong><div class="choices"></div></div>
    <div class="field" data-field="plausible"><strong>Is the pollution visually plausible?</strong><div class="choices"></div></div>
    <label><strong>Notes</strong><textarea id="notes"></textarea></label>
    <div class="nav"><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><span id="counter"></span></div>
  </section>
</main>
<script>
const items=__ITEMS__;
const storageKey='qimg7_image_attack_annotation_v1';
let index=0;
let saved={};
try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
for(const item of items)Object.assign(item,saved[item.row_id]||{});

function image(id,src){const node=document.getElementById(id);node.src=src||'';node.style.display=src?'block':'none'}
function save(){const item=items[index];saved[item.row_id]={on_topic:item.on_topic,fact_flipped:item.fact_flipped,plausible:item.plausible,notes:item.notes};localStorage.setItem(storageKey,JSON.stringify(saved))}
function choose(field,value){items[index][field]=value;save();render()}
function move(delta){index=Math.max(0,Math.min(items.length-1,index+delta));render()}
function renderChoices(){for(const group of document.querySelectorAll('.field')){const field=group.dataset.field;const wrap=group.querySelector('.choices');wrap.innerHTML='';for(const value of ['yes','no','unclear']){const button=document.createElement('button');button.textContent=value[0].toUpperCase()+value.slice(1);button.className='choice'+(items[index][field]===value?' selected':'');button.onclick=()=>choose(field,value);wrap.appendChild(button)}}}
function render(){const item=items[index];document.getElementById('dataset').textContent=item.dataset;document.getElementById('attack').textContent=item.pollution_type;document.getElementById('question').textContent=item.question;document.getElementById('method').textContent=item.manipulation_method||'—';document.getElementById('prompt').textContent=item.manipulation_prompt||'';document.getElementById('original-caption').textContent=item.original_alt||'';document.getElementById('polluted-caption').textContent=item.polluted_alt||'';image('original-image',item.original_image_url);image('polluted-image',item.polluted_image_path||item.polluted_image_url);document.getElementById('notes').value=item.notes||'';document.getElementById('counter').textContent=`Item ${index+1} of ${items.length}`;const done=items.filter(x=>x.on_topic&&x.fact_flipped&&x.plausible).length;document.getElementById('progress').textContent=`${done} / ${items.length} complete`;renderChoices()}
document.getElementById('notes').addEventListener('input',event=>{items[index].notes=event.target.value;save()});
function csvCell(value){const text=String(value??'');return /[",\n]/.test(text)?'"'+text.replaceAll('"','""')+'"':text}
function exportCsv(){const fields=['row_id','dataset','pollution_type','question','on_topic','fact_flipped','plausible','notes'];const rows=[fields.join(','),...items.map(item=>fields.map(field=>csvCell(item[field])).join(','))];const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([rows.join('\n')+'\n'],{type:'text/csv'}));link.download='qimg7_image_attack_annotations.csv';link.click();URL.revokeObjectURL(link.href)}
render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    items = build_items(args.sample_csv, args.repo_root, args.output)
    item_json = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    args.output.write_text(HTML.replace("__ITEMS__", item_json), encoding="utf-8")
    print(f"Wrote {len(items)} frozen annotation items -> {args.output}")


if __name__ == "__main__":
    main()
