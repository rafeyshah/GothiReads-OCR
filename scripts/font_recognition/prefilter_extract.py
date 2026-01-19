#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 helper: Prefilter clean manifest for Option A feasibility.

Option A requires:
  len(pred_units) == len(gt_units)

But CTC can output at most ~T collapsed symbols where T is the time dimension.
So if gt_units_len > T, it is IMPOSSIBLE to pass Option A.

This script filters rows with:
  ok == "TRUE" AND gt_units_len <= max_gt_len

Example:
  python phase2_prefilter_manifest.py \
    --in manifests/train_clean.csv \
    --out manifests/train_clean_T40.csv \
    --max-gt-len 40
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", type=Path, required=True)
    ap.add_argument("--out", dest="out_csv", type=Path, required=True)
    ap.add_argument("--max-gt-len", type=int, required=True,
                    help="Maximum allowed gt_units_len (typically equal to model T).")
    ap.add_argument("--ok-value", type=str, default="TRUE")
    args = ap.parse_args()

    with args.in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "ok" not in fieldnames or "gt_units_len" not in fieldnames:
        raise SystemExit(
            f"Missing required columns in {args.in_csv}. Need ok, gt_units_len")

    kept = []
    for r in rows:
        if r.get("ok") != args.ok_value:
            continue
        try:
            gl = int(r.get("gt_units_len", "0"))
        except Exception:
            continue
        if gl <= args.max_gt_len:
            kept.append(r)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    total = len(rows)
    pct = (len(kept) / total * 100.0) if total else 0.0
    print(
        f"[OK] {args.out_csv}: kept {len(kept)}/{total} ({pct:.2f}%) with max_gt_len={args.max_gt_len}")


if __name__ == "__main__":
    main()
