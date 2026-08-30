#!/usr/bin/env python3
"""
Fill missing pollution records for questions that have fewer than 2 records
of a given type.

Only re-runs the specific missing attacks per question; appends to existing CSV.
Run after pollute_query_images.py has completed at least one full pass.

Usage:
  python fill_gaps.py              # FAVA
  python fill_gaps.py Biography
  python fill_gaps.py --all
"""

import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from pathlib import Path

import pandas as pd

# Reuse all infrastructure from the main script
from pollute_query_images import (
    ALL_DATASETS,
    BASE_DIR,
    EVIDENCE_DIR,
    POLLUTED_CSV_DIR,
    POLLUTED_IMG_DIR,
    N_WORKERS,
    T1_N, T2_N, T3_N, T4_N, T5_N, T6_N, T7_N,
    _gpu_q,
    _gpu_submit,
    _gpu_wait,
    apply_blend,
    best_image_url,
    build_partners,
    download_image,
    gemini_text_params,
    tc_caption_bar,
    tc_watermark,
    t3_gemini_edit,
    vlm_describe,
    _t5_patch_fn,
    _t7_nst_fn,
)


FIELDNAMES = [
    "question_idx", "question", "image_idx", "pollution_type",
    "original_image_url", "original_alt",
    "polluted_image_url", "polluted_alt", "polluted_image_path",
    "manipulation_method", "manipulation_prompt",
    "donor_question_idx", "rationale",
]


def _pick(dl: dict, urls: list):
    """Walk url list until a successful download is found."""
    for url in urls:
        img, raw = dl.get(url, (None, None))
        if img is not None:
            return img, raw
    return None, None


def fill_question(
    qidx: int,
    need: dict,           # {pollution_type: n_missing}
    images: list,
    subject: str,
    partners: list,       # ordered list of partner q_indices
    q_images: dict,
    q_subjects: dict,
    output_imgs_dir: Path,
    next_i: dict,         # {pollution_type: next image_idx to use}
) -> list:
    """Re-run only the missing attacks for one question."""
    question = images[0]["question"]
    records: list = []

    def mkrec(i, ptype, method, prompt, path=None, p_url=None, p_alt=None, donor=None):
        src = images[min(i, len(images) - 1)]
        return {
            "question_idx":        qidx,
            "question":            question,
            "image_idx":           i,
            "pollution_type":      ptype,
            "original_image_url":  src["image_url"],
            "original_alt":        src.get("image_alt", ""),
            "polluted_image_url":  p_url if p_url is not None else src["image_url"],
            "polluted_alt":        p_alt if p_alt is not None else src.get("image_alt", ""),
            "polluted_image_path": path or "",
            "manipulation_method": method,
            "manipulation_prompt": prompt,
            "donor_question_idx":  donor if donor is not None else "",
            "rationale":           f"{ptype} | subject: {subject[:40]} [gap-fill]",
        }

    def save_img(img, prefix: str, i: int) -> str:
        path = output_imgs_dir / f"{prefix}_q{qidx}_i{i}_gf.jpg"
        img.convert("RGB").save(path, "JPEG", quality=88)
        return str(path)

    # Build candidate URLs: primary first, then thumbnails as 2nd-tier fallback
    candidate_urls = []
    for img in images:
        primary = best_image_url(img)
        if primary:
            candidate_urls.append(primary)
    for img in images:
        thumb = img.get("thumbnail_url", "") or ""
        if thumb and thumb not in candidate_urls:
            candidate_urls.append(thumb)

    # Collect donor URLs — try up to 5 partners so T6/T7 have more chances
    MAX_DONOR_TRIES = min(5, len(partners))
    donor_candidates = []
    for p in partners[:MAX_DONOR_TRIES]:
        url = best_image_url(q_images[p][0])
        if url:
            donor_candidates.append((p, url))
        # Also try the partner's thumbnail as fallback
        thumb = q_images[p][0].get("thumbnail_url", "") or ""
        if thumb and thumb != url:
            donor_candidates.append((p, thumb))

    all_urls = list(dict.fromkeys(
        candidate_urls + [url for _, url in donor_candidates]
    ))

    if not all_urls:
        print(f"    [Q{qidx}] no valid URLs — skipping", flush=True)
        return records

    DL_DEADLINE = 20  # hard cap on the whole download phase
    t_dl = time.time()
    print(f"    [Q{qidx}] downloading {len(all_urls)} URLs ...", flush=True)
    dl: dict = {u: (None, None) for u in all_urls}  # default to failure
    pool_dl = ThreadPoolExecutor(max_workers=max(len(all_urls), 1))
    try:
        fm = {pool_dl.submit(download_image, u): u for u in all_urls}
        try:
            for fut in as_completed(fm, timeout=DL_DEADLINE):
                try:
                    dl[fm[fut]] = fut.result()
                except Exception:
                    pass
        except FuturesTimeoutError:
            hung = [u for f, u in fm.items() if not f.done()]
            print(f"    [Q{qidx}] {len(hung)} URLs still hung after {DL_DEADLINE}s — abandoning", flush=True)
    finally:
        pool_dl.shutdown(wait=False, cancel_futures=True)
    n_ok = sum(1 for img, _ in dl.values() if img is not None)
    print(f"    [Q{qidx}] downloads done: {n_ok}/{len(all_urls)} ok  ({time.time()-t_dl:.1f}s)", flush=True)

    if not candidate_urls:
        print(f"    [Q{qidx}] no candidate source URLs — skipping", flush=True)
        return records

    def pick_src(slot: int):
        start = slot % len(candidate_urls)
        for offset in range(len(candidate_urls)):
            url = candidate_urls[(start + offset) % len(candidate_urls)]
            img, raw = dl.get(url, (None, None))
            if img is not None:
                return img, raw
        return None, None

    # Find the first donor that actually downloaded
    good_donor_qidx, good_donor_img = None, None
    for p, url in donor_candidates:
        img, _ = dl.get(url, (None, None))
        if img is not None:
            good_donor_qidx, good_donor_img = p, img
            break

    # Lazily fetch Gemini text params only if needed
    _params = {}
    def params():
        if not _params:
            alts = [img.get("image_alt", "") or subject for img in images[:T1_N]]
            _params.update(gemini_text_params(subject, question, alts))
        return _params

    # ---- T1: Caption Flip ----
    if "caption_flip" in need:
        p = params()
        for k in range(need["caption_flip"]):
            i = next_i.get("caption_flip", 0) + k
            records.append(mkrec(i, "caption_flip", "gemini_flash",
                                 f"subject:{subject}", p_alt=p["captions"][i % T1_N]))

    # ---- T2: Entity Swap ----
    if "entity_swap" in need and partners:
        for k in range(need["entity_swap"]):
            i = next_i.get("entity_swap", 0) + k
            p_idx = partners[i % len(partners)]
            d = q_images[p_idx][i % len(q_images[p_idx])]
            records.append(mkrec(i, "entity_swap", "url_swap", "",
                                 p_url=best_image_url(d),
                                 p_alt=images[i % len(images)].get("image_alt", ""),
                                 donor=p_idx))

    # ---- T3: Semantic Rewrite (parallel) ----
    if "semantic_entity_rewrite" in need:
        p = params()
        indices = [next_i.get("semantic_entity_rewrite", 0) + k
                   for k in range(need["semantic_entity_rewrite"])]

        def _t3_w(i):
            img0, _ = pick_src(i)
            if img0 is None:
                return None
            edited, method, used_prompt = t3_gemini_edit(img0, p["rewrite"])
            return i, method, used_prompt, edited

        with ThreadPoolExecutor(max_workers=len(indices)) as tp:
            futs = {tp.submit(_t3_w, i): i for i in indices}
        for fut, i in futs.items():
            try:
                res = fut.result()
                if res:
                    ri, method, used_prompt, edited = res
                    records.append(mkrec(ri, "semantic_entity_rewrite", method,
                                         used_prompt, save_img(edited, "T3", ri)))
            except Exception as exc:
                print(f"  [T3 gap] Q{qidx} i{i}: {exc}", flush=True)

    # ---- T4: FigStep Typography ----
    if "figstep_typography" in need:
        p = params()
        for k in range(need["figstep_typography"]):
            i = next_i.get("figstep_typography", 0) + k
            img0, _ = pick_src(i)
            if img0 is None:
                continue
            try:
                style = i % 2
                attacked = tc_caption_bar(img0, p["figstep"]) if style == 0 else tc_watermark(img0, p["figstep"])
                label = "caption_bar" if style == 0 else "watermark"
                records.append(mkrec(i, "figstep_typography", f"pil_{label}",
                                     p["figstep"], save_img(attacked, "T4", i)))
            except Exception as exc:
                print(f"  [T4 gap] Q{qidx} i{i}: {exc}", flush=True)

    # ---- T5: Adversarial Patch ----
    if "adversarial_patch" in need:
        p = params()
        tickets = []
        indices = [next_i.get("adversarial_patch", 0) + k
                   for k in range(need["adversarial_patch"])]
        for i in indices:
            img0, _ = pick_src(i)
            if img0 is not None:
                tickets.append((i, _gpu_submit(_t5_patch_fn, img0, p["clip_concept"])))
        for i, ticket in tickets:
            try:
                attacked = _gpu_wait(ticket)
                records.append(mkrec(i, "adversarial_patch", "pgd_patch_clip_vitl14",
                                     p["clip_concept"], save_img(attacked, "T5", i)))
            except Exception as exc:
                print(f"  [T5 gap] Q{qidx} i{i}: {exc}", flush=True)

    # ---- T6: Image Blend ----
    if "image_blend" in need:
        orig_img, orig_bytes = pick_src(0)
        if orig_img is None or good_donor_img is None:
            print(f"    [T6 Q{qidx}] skip — src_ok={orig_img is not None} donor_ok={good_donor_img is not None}", flush=True)
        else:
            print(f"    [T6 Q{qidx}] calling vlm_describe ...", flush=True)
            t_vlm = time.time()
            vlm_desc = vlm_describe(orig_bytes, subject) if orig_bytes else ""
            print(f"    [T6 Q{qidx}] vlm done ({time.time()-t_vlm:.1f}s): '{vlm_desc[:40]}'", flush=True)
            for k in range(need["image_blend"]):
                i = next_i.get("image_blend", 0) + k
                try:
                    composite = apply_blend(orig_img, good_donor_img)
                    donor_subj = q_subjects.get(good_donor_qidx, "unknown")
                    records.append(mkrec(i, "image_blend", "vlm_pil_composite",
                                         f"VLM:{vlm_desc}|donor:{donor_subj}",
                                         save_img(composite, "T6", i),
                                         donor=good_donor_qidx))
                    print(f"    [T6 Q{qidx}] i{i} blended ok", flush=True)
                except Exception as exc:
                    print(f"    [T6 Q{qidx}] i{i}: {exc}", flush=True)

    # ---- T7: Neural Style Transfer ----
    if "neural_style_transfer" in need:
        if good_donor_img is None:
            print(f"  [T7 gap skip] Q{qidx} — no usable donor for style", flush=True)
        else:
            tickets = []
            indices = [next_i.get("neural_style_transfer", 0) + k
                       for k in range(need["neural_style_transfer"])]
            for i in indices:
                img0, _ = pick_src(i)
                if img0 is not None:
                    tickets.append((i, _gpu_submit(_t7_nst_fn, img0, good_donor_img)))
            donor_subj = q_subjects.get(good_donor_qidx, "unknown")
            for i, ticket in tickets:
                try:
                    styled = _gpu_wait(ticket)
                    records.append(mkrec(i, "neural_style_transfer", "gram_nst_vgg19",
                                         f"style_donor:Q{good_donor_qidx}|{donor_subj[:30]}",
                                         save_img(styled, "T7", i),
                                         donor=good_donor_qidx))
                except Exception as exc:
                    print(f"  [T7 gap] Q{qidx} i{i}: {exc}", flush=True)

    return records


def fill_dataset(dataset: str) -> None:
    input_csv       = EVIDENCE_DIR     / f"{dataset}_query_image_evidence.csv"
    output_csv      = POLLUTED_CSV_DIR / f"{dataset}_query_image_polluted.csv"
    output_imgs_dir = POLLUTED_IMG_DIR / dataset

    if not output_csv.exists():
        print(f"[SKIP] {dataset}: no output CSV yet — run main script first")
        return
    if not input_csv.exists():
        print(f"[SKIP] {dataset}: no input CSV")
        return

    print(f"\n{'='*60}")
    print(f"=== Gap-fill — {dataset} ===")

    # Load existing results
    existing = pd.read_csv(output_csv)
    all_types = ["caption_flip", "entity_swap", "semantic_entity_rewrite",
                 "figstep_typography", "adversarial_patch", "image_blend",
                 "neural_style_transfer"]

    # Find gaps
    gaps: dict = {}    # qidx -> {type: n_missing}
    next_i: dict = {}  # qidx -> {type: max_i + 1}
    for qidx, grp in existing.groupby("question_idx"):
        by_type = grp.groupby("pollution_type")["image_idx"].agg(["count", "max"])
        q_need = {}
        q_next = {}
        for t in all_types:
            if t in by_type.index:
                cnt = int(by_type.loc[t, "count"])
                mx  = int(by_type.loc[t, "max"])
            else:
                cnt, mx = 0, -1
            if cnt < 2:
                q_need[t] = 2 - cnt
                q_next[t] = mx + 1
        if q_need:
            gaps[int(qidx)] = q_need
            next_i[int(qidx)] = q_next

    if not gaps:
        print("  No gaps found.")
        return

    total_missing = sum(sum(v.values()) for v in gaps.values())
    print(f"  {len(gaps)} questions with gaps, {total_missing} missing records")

    # Load input CSV
    img_df = pd.read_csv(input_csv)
    q_images: dict   = {}
    q_subjects: dict = {}
    for _, row in img_df.iterrows():
        qidx = int(row["question_idx"])
        q_images.setdefault(qidx, []).append(row.to_dict())
        if qidx not in q_subjects:
            q_subjects[qidx] = str(row.get("question", ""))[:50].rstrip(".,?! ")

    q_indices = sorted(q_images.keys())
    swap_partners = build_partners(q_indices, q_subjects)

    output_imgs_dir.mkdir(parents=True, exist_ok=True)
    csv_lock  = threading.Lock()
    completed = [0]
    t0        = time.time()

    with open(output_csv, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)

        def work(qidx: int) -> None:
            imgs     = q_images.get(qidx, [])
            if not imgs:
                return
            subject  = q_subjects.get(qidx, "")
            partners = swap_partners.get(qidx, [])
            need     = gaps[qidx]
            ni       = next_i[qidx]

            print(f"  [gap] Q{qidx} | {subject[:40]} | need {need}", flush=True)
            recs = fill_question(qidx, need, imgs, subject, partners,
                                 q_images, q_subjects, output_imgs_dir, ni)

            with csv_lock:
                writer.writerows(recs)
                fout.flush()
                completed[0] += 1
                n   = completed[0]
                ela = time.time() - t0
                print(f"  [{n}/{len(gaps)}] Q{qidx} — filled {len(recs)} records "
                      f"({ela/n:.0f}s/Q)", flush=True)

        Q_TIMEOUT = 180  # seconds; a question should never take more than 3 min
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(work, q): q for q in gaps}
            for fut in as_completed(futs, timeout=Q_TIMEOUT * len(gaps)):
                if (exc := fut.exception()):
                    print(f"  [worker error] Q{futs[fut]}: {exc}", flush=True)

    # Final count
    final = pd.read_csv(output_csv)
    still_missing = []
    for qidx, grp in final.groupby("question_idx"):
        by_type = grp["pollution_type"].value_counts()
        for t in all_types:
            if by_type.get(t, 0) < 2:
                still_missing.append((qidx, t, by_type.get(t, 0)))
    if still_missing:
        print(f"\n  Still missing ({len(still_missing)} slots):")
        for qidx, t, cnt in still_missing[:10]:
            print(f"    Q{qidx} {t}: {cnt}/2")
    else:
        print(f"\n  All gaps filled. Total records: {len(final)}")


def main() -> None:
    POLLUTED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:]
    if "--all" in args:
        datasets = ALL_DATASETS
    else:
        ds = next((a for a in args if not a.startswith("--")), "FAVA")
        datasets = [ds]

    for ds in datasets:
        fill_dataset(ds)

    _gpu_q.put(None)


if __name__ == "__main__":
    main()
