#!/usr/bin/env python3
"""Emit a per-image manifest and a reconciled count report for the QIMG-7 image tree.

The manifest carries one row per file under ``--image_root/images``:

* provenance joined from ``benchmark/pool/*.csv`` (dataset, question, attack
  family, source URL, manipulation method);
* a SHA-256 content hash and byte size; and
* empty rights columns for human completion.

The rights columns are deliberately left blank. A license determination is a
human judgement, and a machine-guessed one is worse than an acknowledged gap.

Hashes are read from the Git LFS pointer when a file has not been materialised,
so the manifest is exact on a checkout without ``git lfs pull``.
"""

import argparse
import csv
import glob
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict

POOL_GLOB = "*_query_image_polluted.csv"
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"
LFS_OID = re.compile(rb"^oid sha256:([0-9a-f]{64})$", re.M)
LFS_SIZE = re.compile(rb"^size (\d+)$", re.M)
ORIG_NAME = re.compile(r"^orig_q(\d+)_i(\d+)\.")

RIGHTS_FIELDS = [
    "rightsholder",
    "source_license",
    "source_license_url",
    "derivative_permission",
    "commercial_restriction",
    "verification_date",
    "include_decision",
]

FIELDS = [
    "path",
    "dataset",
    "role",
    "question_idx",
    "image_idx",
    "pollution_type",
    "manipulation_method",
    "original_image_url",
    "sha256",
    "bytes",
    "hash_source",
] + RIGHTS_FIELDS


def hash_file(path):
    """Return (sha256, size, source) for a file, honouring unfetched LFS pointers."""
    with open(path, "rb") as fh:
        head = fh.read(1024)
    if head.startswith(LFS_MAGIC):
        oid = LFS_OID.search(head)
        size = LFS_SIZE.search(head)
        if oid:
            return (
                oid.group(1).decode(),
                int(size.group(1)) if size else "",
                "lfs_pointer",
            )
        return "", "", "lfs_pointer_unparsed"
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total, "file_bytes"


def load_pool(pool_dir):
    """Index pool rows by polluted image path and by (dataset, qid, image slot)."""
    by_path, by_slot, rows_with_local = {}, {}, 0
    for csv_path in sorted(glob.glob(os.path.join(pool_dir, POOL_GLOB))):
        dataset = os.path.basename(csv_path).split("_query_image_polluted")[0]
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                slot = (dataset, row["question_idx"], row["image_idx"])
                if row.get("original_image_url"):
                    by_slot.setdefault(slot, row["original_image_url"])
                local = (row.get("polluted_image_path") or "").strip()
                if not local:
                    continue
                rows_with_local += 1
                by_path.setdefault(local, dict(row, dataset=dataset))
    return by_path, by_slot, rows_with_local


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool_dir", default="benchmark/pool")
    ap.add_argument("--image_root", default=".",
                    help="Directory containing the images/ tree (pool paths are "
                         "relative to it).")
    ap.add_argument("--out", default="metadata/qimg7_image_manifest.csv")
    ap.add_argument("--counts_only", action="store_true",
                    help="Print the reconciliation report without writing a manifest.")
    args = ap.parse_args()

    for required in (args.pool_dir, os.path.join(args.image_root, "images")):
        if not os.path.isdir(required):
            sys.exit(f"error: required directory not found: {required}")

    by_path, by_slot, rows_with_local = load_pool(args.pool_dir)
    if not by_path:
        sys.exit(f"error: no pool rows with a local image path under {args.pool_dir}")

    records = []
    for root, _, files in os.walk(os.path.join(args.image_root, "images")):
        for name in sorted(files):
            if name.startswith("."):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, args.image_root)
            pool_row = by_path.get(rel)
            dataset = rel.split(os.sep)[1] if os.sep in rel else ""
            sha, size, source = hash_file(full)
            rec = dict.fromkeys(FIELDS, "")
            rec.update(path=rel, dataset=dataset, sha256=sha, bytes=size,
                       hash_source=source)
            if pool_row:
                rec.update(
                    role="attack_derivative",
                    question_idx=pool_row["question_idx"],
                    image_idx=pool_row["image_idx"],
                    pollution_type=pool_row["pollution_type"],
                    manipulation_method=pool_row.get("manipulation_method", ""),
                    original_image_url=pool_row.get("original_image_url", ""),
                )
            else:
                rec["role"] = "clean_original" if ORIG_NAME.match(name) else "unreferenced"
                match = ORIG_NAME.match(name)
                if match:
                    qid, slot = match.group(1), match.group(2)
                    rec.update(question_idx=qid, image_idx=slot,
                               original_image_url=by_slot.get((dataset, qid, slot), ""))
            records.append(rec)

    on_disk = {r["path"] for r in records}
    missing = sorted(p for p in by_path if p not in on_disk)
    hashes = Counter(r["sha256"] for r in records if r["sha256"])
    dup_groups = {h: n for h, n in hashes.items() if n > 1}
    roles = Counter(r["role"] for r in records)
    per_attack = Counter(r["pollution_type"] for r in records if r["pollution_type"])

    print(f"files under images/                     {len(records)}")
    for role in ("attack_derivative", "clean_original", "unreferenced"):
        print(f"  role={role:<22s}             {roles.get(role, 0)}")
    print(f"distinct attack image paths in pool     {len(by_path)}")
    print(f"pool records citing a local image       {rows_with_local}")
    print(f"pool image paths missing from disk      {len(missing)}")
    print(f"unfetched LFS pointers                  "
          f"{sum(1 for r in records if r['hash_source'] == 'lfs_pointer')}")
    print(f"unique sha256 values                    {len(hashes)}")
    print(f"byte-identical duplicate groups         {len(dup_groups)} "
          f"({sum(n - 1 for n in dup_groups.values())} redundant files)")
    if per_attack:
        print("attack images by family:")
        for attack, count in sorted(per_attack.items()):
            print(f"  {attack:<26s} {count}")
    for path in missing[:10]:
        print(f"  MISSING: {path}", file=sys.stderr)

    if args.counts_only:
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: r["path"]))
    print(f"\nwrote {len(records)} rows to {args.out}")
    print("Rights columns are intentionally empty and require human completion: "
          + ", ".join(RIGHTS_FIELDS))


if __name__ == "__main__":
    main()
