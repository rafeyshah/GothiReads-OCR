#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import time
import argparse
from pathlib import Path
import yaml

from typing import List

from paddleocr import PaddleOCR
from harness import load_val_split, evaluate, cer_per_line


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--rec_model_dir", required=True,
                    help="Exported inference dir containing inference.pdmodel etc.")
    ap.add_argument("--rec_char_dict_path", required=True,
                    help="Your gothi_dict.txt used in training/export")
    ap.add_argument("--max_text_length", type=int, default=50)
    return ap.parse_args()


def load_dict_from_infer_cfg(rec_model_dir: Path):
    """
    Load the true character dictionary from an exported PaddleOCR inference model.
    """
    rec_model_dir = Path(rec_model_dir)

    # Try common names
    for name in ["infer_cfg.yml", "inference.yml", "inference (1).yml"]:
        p = rec_model_dir / name
        if p.exists():
            infer_cfg = p
            break
    else:
        raise FileNotFoundError(f"No infer_cfg.yml found in {rec_model_dir}")

    y = yaml.safe_load(infer_cfg.read_text(encoding="utf-8"))
    char_list = y["PostProcess"]["character_dict"]

    # Normalize to strings
    char_list = ["" if c is None else str(c) for c in char_list]

    # Write a stable dict file next to the model
    out_dict = rec_model_dir / "character_dict_from_infer_cfg.txt"
    if not out_dict.exists():
        out_dict.write_text("\n".join(char_list) + "\n", encoding="utf-8")

    return out_dict


def init_ocr(args) -> PaddleOCR:
    # Pure PaddleOCR inference: no PaddleX, no torch.
    # det=False, cls=False because you already have line crops.
    return PaddleOCR(
        use_gpu=bool(args.use_gpu),
        use_angle_cls=False,
        det=False,
        rec=True,
        rec_model_dir=args.rec_model_dir,
        rec_char_dict_path=args.rec_char_dict_path,
        max_text_length=args.max_text_length,
        show_log=False,
    )


def recognize_all(ocr: PaddleOCR, img_paths: List[str]) -> List[str]:
    preds: List[str] = []
    for p in img_paths:
        res = ocr.ocr(p, det=False, cls=False)
        # res usually: [ [ ['TEXT', score], ... ] ]
        if not res or not res[0]:
            preds.append("")
            continue
        first = res[0][0]
        if isinstance(first, (list, tuple)) and len(first) >= 1 and isinstance(first[0], str):
            preds.append(first[0])
        else:
            preds.append(str(first))
    return preds


def main():
    args = parse_args()
    # IMPORTANT: Always use dict embedded in exported inference model
    args.rec_char_dict_path = str(
        load_dict_from_infer_cfg(Path(args.rec_model_dir))
    )

    ids, img_paths, gts = load_val_split(args.manifest, limit=args.limit)
    if not ids:
        raise RuntimeError(
            f"No valid lines found in manifest: {args.manifest}")

    ocr = init_ocr(args)

    t0 = time.perf_counter()
    preds = recognize_all(ocr, img_paths)
    t1 = time.perf_counter()

    metrics = evaluate(preds, gts, ids)
    metrics["model"] = {
        "engine": "PaddleOCR",
        "rec_model_dir": args.rec_model_dir,
        "rec_char_dict_path": args.rec_char_dict_path,
        "det": False,
        "cls": False,
    }
    metrics["val_manifest"] = str(Path(args.manifest))
    metrics["runtime_seconds"] = t1 - t0
    metrics["avg_latency_ms_per_line"] = (t1 - t0) * 1000.0 / max(1, len(ids))
    metrics["num_lines"] = len(ids)

    if args.out_dir:
        _toggle = Path(args.out_dir)
    else:
        from datetime import datetime
        out_dir = Path("runs") / "ppocrv5_pure" / \
            datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "preds.txt").open("w", encoding="utf-8") as f:
        for i, p in zip(ids, preds):
            f.write(f"{i}\t{p}\n")

    with (out_dir / "per_line.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_id", "gt", "pred", "CER"])
        for i, g, p in zip(ids, gts, preds):
            w.writerow([i, g, p, cer_per_line(g, p)])

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Saved results to", out_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
