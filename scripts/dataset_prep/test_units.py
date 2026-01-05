#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny sanity test: sample 20 random items from a manifest and assert
len(graphemes(txt_clean)) == len(font_tokens(font_clean)).

Usage:
  python test_units.py --manifest manifests/train.csv
  python test_units.py --manifest manifests/valid_clean.csv --n 50 --seed 123

Notes:
  - The manifest is expected to have txt_path and font_path columns (like build_manifest.py outputs).
  - Rows with missing files or ok != TRUE are skipped by default.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from units import graphemes, font_tokens, normalize, strip_ws


def read_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include_not_ok", action="store_true", help="Also sample rows where ok != TRUE")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    rows = []
    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if not args.include_not_ok and r.get("ok") != "TRUE":
                continue
            txt_p = Path(r.get("txt_path", "")).expanduser()
            font_p = Path(r.get("font_path", "")).expanduser()
            if not (txt_p.exists() and font_p.exists()):
                continue
            rows.append((r.get("id", ""), txt_p, font_p))

    if not rows:
        raise SystemExit("No usable rows found (check paths and ok==TRUE rows).")

    rnd = random.Random(args.seed)
    sample = rows if len(rows) <= args.n else rnd.sample(rows, args.n)

    failures = 0
    for img_id, txt_p, font_p in sample:
        txt_raw = read_file(txt_p).rstrip("\n")
        font_raw = read_file(font_p)

        # Apply canonical policy
        txt_units = graphemes(txt_raw)
        font_units = font_tokens(font_raw, expected_len=len(txt_units))

        if len(txt_units) != len(font_units):
            failures += 1
            print("FAIL", img_id)
            print("  txt_path :", txt_p)
            print("  font_path:", font_p)
            print("  len(txt_units) =", len(txt_units))
            print("  len(font_units)=", len(font_units))
            # show a small preview (safe)
            print("  txt_preview :", strip_ws(normalize(txt_raw))[:80])
            print("  font_preview:", strip_ws(normalize(font_raw))[:80])
            print()

    if failures:
        raise SystemExit(f"{failures}/{len(sample)} samples FAILED length match.")
    print(f"OK: {len(sample)}/{len(sample)} samples passed length match.")


if __name__ == "__main__":
    main()
