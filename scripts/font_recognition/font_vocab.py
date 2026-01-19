#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-jsonl", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output vocab json")
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()

    cnt = Counter()
    total_lines = 0
    total_g = 0

    for p in args.align_jsonl:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                ex = json.loads(line)
                if not ex.get("ok_align"):
                    continue
                fonts = ex.get("gt_fonts")
                if not isinstance(fonts, list):
                    continue
                for lab in fonts:
                    cnt[str(lab)] += 1
                total_g += len(fonts)

    labels = [lab for lab, c in cnt.most_common() if c >= args.min_count]
    font2id = {lab: i for i, lab in enumerate(labels)}
    out = {
        "num_fonts": len(labels),
        "labels": labels,
        "font2id": font2id,
        "counts": {lab: int(cnt[lab]) for lab in labels},
        "total_graphemes": int(total_g),
        "total_lines_seen": int(total_lines),
        "min_count": int(args.min_count),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[OK] wrote {args.out} num_fonts={len(labels)} total_graphemes={total_g}")


if __name__ == "__main__":
    main()
