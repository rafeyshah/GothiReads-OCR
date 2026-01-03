#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build FONT-clean manifests where GT units match .font units.

Why you need this:
- Using Unicode grapheme clusters (\X) makes "p̃" count as 1 unit.
- But PaddleOCR CTC + .font typically treat "p" and "̃" separately (2 units).
- So for font supervision we must count GT as Unicode codepoints (Python characters),
  usually dropping whitespace.

This script builds:
  {split}_font_clean.csv
with columns:
  split,id,image_path,txt_path,font_path,num_units,num_fonts,ok

ok == TRUE only when:
  len(gt_units_for_font(gt_text)) == len(font_labels)

Example:
python scripts/build_manifest_font.py \
  --data-root /content/dataset \
  --splits train val test \
  --out-dir /content/manifests/font \
  --drop-spaces
"""

import argparse
import csv
import sys
from pathlib import Path

import regex as re

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace").rstrip("\n")


def read_font_labels(p: Path):
    """
    .font semantics:
    - Remove whitespace
    - Remaining characters are per-unit font labels
    """
    raw = p.read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"\s+", "", raw)
    return list(raw)


def gt_units_for_font(text: str, drop_spaces: bool = True):
    """
    Codepoints (Python characters), optionally dropping whitespace.
    Combining marks count separately => matches .font and CTC behavior.
    """
    units = list(text)
    if drop_spaces:
        units = [u for u in units if not u.isspace()]
    return units


def id_key(split_root: Path, p: Path) -> str:
    rel = p.relative_to(split_root)
    return str(rel.with_suffix("")).replace("\\", "/")


def discover_triples(split_root: Path):
    images, texts, fonts = {}, {}, {}

    for p in split_root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        k = id_key(split_root, p)

        if ext in IMG_EXTS:
            images[k] = p
        elif ext == ".txt":
            texts[k] = p
        elif ext == ".font":
            fonts[k] = p

    triples = []
    all_keys = set(images) | set(texts) | set(fonts)
    for k in sorted(all_keys):
        triples.append((k, images.get(k), texts.get(k), fonts.get(k)))
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root directory containing split folders (train/, val/, test/).")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    help="Split folder names under data-root.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to write manifest CSVs.")
    ap.add_argument("--drop-spaces", action="store_true", default=True,
                    help="Drop whitespace codepoints from GT when computing units.")
    ap.add_argument("--fail-if-missing", action="store_true",
                    help="Exit non-zero if any item is missing an image/txt/font.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall_missing = 0

    for split in args.splits:
        split_root = args.data_root / split
        if not split_root.exists():
            print(
                f"[WARN] Split folder not found: {split_root}", file=sys.stderr)
            continue

        triples = discover_triples(split_root)
        out_csv = args.out_dir / f"{split}_font_clean.csv"

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split", "id", "image_path", "txt_path", "font_path",
                    "num_units", "num_fonts", "ok"
                ],
            )
            writer.writeheader()

            for stem, img_p, txt_p, font_p in triples:
                rec = {
                    "split": split,
                    "id": stem,
                    "image_path": str(img_p) if img_p else "",
                    "txt_path": str(txt_p) if txt_p else "",
                    "font_path": str(font_p) if font_p else "",
                    "num_units": "",
                    "num_fonts": "",
                    "ok": ""
                }

                missing = (img_p is None) or (
                    txt_p is None) or (font_p is None)
                if missing:
                    overall_missing += 1
                    rec["ok"] = "MISSING"
                    writer.writerow(rec)
                    continue

                gt_text = read_text(txt_p)
                gt_units = gt_units_for_font(
                    gt_text, drop_spaces=args.drop_spaces)
                font_labels = read_font_labels(font_p)

                rec["num_units"] = len(gt_units)
                rec["num_fonts"] = len(font_labels)
                rec["ok"] = "TRUE" if len(gt_units) == len(
                    font_labels) else "LEN_MISMATCH"
                writer.writerow(rec)

        print(f"[OK] Wrote {out_csv} ({len(triples)} rows)")

    if args.fail_if_missing and overall_missing > 0:
        print(
            f"[ERROR] Missing files found across splits: {overall_missing}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
