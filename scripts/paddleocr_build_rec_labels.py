#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build PaddleOCR-style rec label files from Gothi-Read manifests.

Usage:
  python scripts/paddleocr_build_rec_labels.py \
      --manifest /content/manifests/train_clean.csv \
      --out /content/gothiread_paddle/train_list.txt
"""

import argparse
from pathlib import Path

from harness import load_val_split  # from your day3 harness


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    ids, img_paths, gts = load_val_split(args.manifest, limit=args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for img, gt in zip(img_paths, gts):
            # PaddleOCR expects "path<TAB>text"
            f.write(f"{img}\t{gt}\n")

    print(f"Wrote {len(img_paths)} lines to {out_path}")


if __name__ == "__main__":
    main()
