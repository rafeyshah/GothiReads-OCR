#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build manifest CSV with canonical unit pipeline (units_v2.py).

Outputs columns:
split,id,image_path,txt_path,font_path,gt_units_len,font_units_len,ok,reason

ok values:
- TRUE
- MISSING
- READ_ERROR
- PARSE_ERROR
- LEN_MISMATCH

Example:
  python build_manifest_v3.py --data-root /content/dataset --splits train valid test --out-dir manifests
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from units import graphemes, font_tokens

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def id_key(split_root: Path, p: Path) -> str:
    rel = p.relative_to(split_root)
    return str(rel.with_suffix("")).replace("\\", "/")


def discover_triples(split_root: Path):
    images, texts, fonts = {}, {}, {}

    for p in split_root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        key = id_key(split_root, p)
        if ext in IMG_EXTS:
            images[key] = p
        elif ext == ".txt":
            texts[key] = p
        elif ext == ".font":
            fonts[key] = p

    triples = []
    all_keys = set(images) | set(texts) | set(fonts)
    for k in sorted(all_keys):
        triples.append((k, images.get(k), texts.get(k), fonts.get(k)))
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root data dir containing split folders (e.g., train/, valid/, test/).")
    ap.add_argument("--splits", nargs="+", default=["train", "valid"],
                    help="Split folder names under data-root.")
    ap.add_argument("--out-dir", type=Path, default=Path("manifests"),
                    help="Where to write CSVs.")
    ap.add_argument("--fail-if-missing", action="store_true",
                    help="Exit non-zero if any item is missing .txt/.font/image.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall_missing = 0

    for split in args.splits:
        split_root = args.data_root / split
        if not split_root.exists():
            print(f"[WARN] Split folder not found: {split_root}", file=sys.stderr)
            continue

        triples = discover_triples(split_root)
        out_csv = args.out_dir / f"{split}.csv"

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split", "id", "image_path", "txt_path", "font_path",
                    "gt_units_len", "font_units_len", "ok", "reason",
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
                    "gt_units_len": "",
                    "font_units_len": "",
                    "ok": "",
                    "reason": "",
                }

                if img_p is None or txt_p is None or font_p is None:
                    overall_missing += 1
                    rec["ok"] = "MISSING"
                    rec["reason"] = "image/txt/font missing"
                    writer.writerow(rec)
                    continue

                try:
                    txt_raw = read_text(txt_p)
                    font_raw = read_text(font_p)
                except Exception as e:
                    rec["ok"] = "READ_ERROR"
                    rec["reason"] = f"read error: {type(e).__name__}"
                    writer.writerow(rec)
                    continue

                try:
                    gt_units = graphemes(txt_raw)
                    ft_units = font_tokens(font_raw, expected_len=len(gt_units))
                except Exception as e:
                    rec["ok"] = "PARSE_ERROR"
                    rec["reason"] = f"parse error: {type(e).__name__}"
                    writer.writerow(rec)
                    continue

                rec["gt_units_len"] = len(gt_units)
                rec["font_units_len"] = len(ft_units)

                if len(gt_units) == len(ft_units):
                    rec["ok"] = "TRUE"
                    rec["reason"] = ""
                else:
                    rec["ok"] = "LEN_MISMATCH"
                    rec["reason"] = "len(graphemes(txt_clean)) != len(font_tokens(font_clean))"

                writer.writerow(rec)

        print(f"[OK] Wrote {out_csv} ({len(triples)} rows)")

    if args.fail_if_missing and overall_missing > 0:
        print(f"[ERROR] Missing files found across splits: {overall_missing}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
