#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter manifests to keep only clean/aligned rows.

Default: keep ok == "TRUE"

Example:
  python filter_clean_v3.py --manifests manifests/train.csv manifests/valid.csv --out-dir manifests --suffix _clean
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def filter_one(in_csv: Path, out_csv: Path, ok_value: str = "TRUE") -> dict:
    kept = 0
    total = 0
    with in_csv.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        if "ok" not in fieldnames:
            raise ValueError(f"{in_csv} has no 'ok' column")

        rows = list(reader)

    total = len(rows)
    kept_rows = [r for r in rows if r.get("ok") == ok_value]
    kept = len(kept_rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return {"input": str(in_csv), "output": str(out_csv), "kept": kept, "total": total, "pct": (kept / total * 100.0) if total else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", type=Path, required=True, help="Input manifest CSVs.")
    ap.add_argument("--out-dir", type=Path, default=Path("manifests"), help="Output directory.")
    ap.add_argument("--suffix", type=str, default="_clean", help="Suffix added before .csv")
    ap.add_argument("--ok-value", type=str, default="TRUE", help="Value in 'ok' column to keep.")
    args = ap.parse_args()

    for in_csv in args.manifests:
        out_csv = args.out_dir / (in_csv.stem + args.suffix + in_csv.suffix)
        stats = filter_one(in_csv, out_csv, ok_value=args.ok_value)
        print(f"[OK] {in_csv.name}: kept {stats['kept']}/{stats['total']} ({stats['pct']:.2f}%) -> {out_csv}")

if __name__ == "__main__":
    main()
