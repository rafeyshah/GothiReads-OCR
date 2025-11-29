#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-shot PARSeq evaluation for Gothi-Read (ICDAR 2024 Track B)

- Uses PARSeq via torch.hub: baudm/parseq
- Assumes input images are already line crops (no detection).
- Reads standard manifest CSV (id, image_path, txt_path, ok).
- Runs on GPU if available (Colab T4 etc.), optional fp16.
- Outputs:
    - preds.txt   (img_id \\t pred)
    - perline.csv (img_id, gt, pred, CER)
    - metrics.json (CER, WER, timing, etc.)

Example usage (on Colab):

    !python /content/GothiRead/scripts/zeroshot/zeroshot_parseq.py \\
        --manifest /content/manifests/valid_clean.csv \\
        --model parseq \\
        --batch_size 32 \\
        --max_length 32

For quick tests:

    --limit 500
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from PIL import Image
from torchvision import transforms as T


# ---------------------------------------------------------------------
# Manifest + metrics utils (similar style to zeroshot_donut.py)
# ---------------------------------------------------------------------

def load_manifest(csv_path: str, limit: Optional[int] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Load id, image_path, txt_path from manifest CSV.
    Uses rows where ok is truthy and paths exist (relative to manifest dir if needed).
    """
    ids: List[str] = []
    img_paths: List[str] = []
    gts: List[str] = []

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit is not None and len(ids) >= limit:
                break

            ok_val = str(row.get("ok", "")).strip().upper()
            if ok_val not in {"TRUE", "1", "YES", "Y"}:
                continue

            img = row.get("image_path") or row.get("img_path")
            txt = row.get("txt_path")
            if not img or not txt:
                continue

            img_path = Path(img)
            if not img_path.is_file():
                img_path = (csv_path.parent / img_path).resolve()
                if not img_path.is_file():
                    continue

            txt_path = Path(txt)
            if not txt_path.is_file():
                txt_path = (csv_path.parent / txt_path).resolve()
                if not txt_path.is_file():
                    continue

            try:
                gt = txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                continue

            img_id = row.get("id") or img_path.stem

            ids.append(img_id)
            img_paths.append(str(img_path))
            gts.append(gt)

    if not ids:
        raise RuntimeError(f"No valid rows found in manifest: {csv_path}")

    return ids, img_paths, gts


def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance (char-level)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        ca = a[i - 1]
        for j in range(1, n + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def compute_cer(gts: List[str], preds: List[str]) -> float:
    total_edits = 0
    total_chars = 0
    for gt, pred in zip(gts, preds):
        total_edits += _edit_distance(gt, pred)
        total_chars += max(1, len(gt))
    return total_edits / total_chars if total_chars > 0 else 0.0


def compute_wer(gts: List[str], preds: List[str]) -> float:
    def words(s: str) -> List[str]:
        return s.strip().split()

    total_edits = 0
    total_words = 0
    for gt, pred in zip(gts, preds):
        gt_words = words(gt)
        pred_words = words(pred)
        total_words += max(1, len(gt_words))

        m, n = len(gt_words), len(pred_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if gt_words[i - 1] == pred_words[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        total_edits += dp[m][n]

    return total_edits / total_words if total_words > 0 else 0.0


def cer_per_line(gt: str, pred: str) -> float:
    return _edit_distance(gt, pred) / max(1, len(gt))


# ---------------------------------------------------------------------
# PARSeq inference
# ---------------------------------------------------------------------

def get_preprocess(image_height: int = 32, image_width: int = 128) -> T.Compose:
    """
    Standard PARSeq preprocessing:
    - Resize to (H, W) with bicubic interpolation
    - ToTensor
    - Normalize mean=0.5, std=0.5
    """
    return T.Compose([
        T.Resize((image_height, image_width), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])


def run_parseq(
    model_name: str,
    ids: List[str],
    img_paths: List[str],
    gts: List[str],
    batch_size: int = 32,
    max_length: int = 32,
    image_height: int = 32,
    image_width: int = 128,
    force_cpu: bool = False,
    fp16: bool = False,
) -> Dict[str, Any]:
    """
    Zero-shot PARSeq inference.
    model_name: one of ['parseq', 'parseq_tiny', 'abinet', 'crnn', 'trba', 'vitstr', ...]
                as supported by baudm/parseq hub.
    """
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading PARSeq model from torch.hub: {model_name}")
    model = torch.hub.load("baudm/parseq", model_name, pretrained=True)
    model.eval()
    model.to(device)

    if device.type == "cuda" and fp16:
        model.half()
        print("Using fp16 on GPU")

    preprocess = get_preprocess(
        image_height=image_height, image_width=image_width)

    preds: List[str] = []

    print(f"Using device: {device}")
    print(
        f"Total lines: {len(ids)}, batch_size={batch_size}, "
        f"max_length={max_length}, size=({image_height},{image_width})"
    )

    start_time = time.time()

    with torch.no_grad():
        for start_idx in range(0, len(ids), batch_size):
            end_idx = min(len(ids), start_idx + batch_size)
            batch_paths = img_paths[start_idx:end_idx]

            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))

            batch_tensor = torch.stack(images, dim=0)  # (B, C, H, W)
            batch_tensor = batch_tensor.to(device)

            if device.type == "cuda" and fp16:
                batch_tensor = batch_tensor.half()

            # logits: (B, T, vocab_size)
            logits = model(batch_tensor, max_length=max_length)
            probs = logits.softmax(-1)

            # tokenizer.decode returns (labels, probs)
            labels, _ = model.tokenizer.decode(probs)
            # labels is a list of strings
            preds.extend([t.strip() for t in labels])

            print(f"Processed {end_idx}/{len(ids)} lines",
                  end="\r", flush=True)

            del images, batch_tensor, logits, probs, labels
            if device.type == "cuda":
                torch.cuda.empty_cache()

    total_time = time.time() - start_time
    avg_sec_per_line = total_time / len(ids)
    lines_per_sec = len(ids) / total_time if total_time > 0 else 0.0

    cer = compute_cer(gts, preds)
    wer = compute_wer(gts, preds)

    print()
    print(f"Done PARSeq inference. CER={cer:.6f}, WER={wer:.6f}")
    print(f"Total time: {total_time:.2f}s  "
          f"({avg_sec_per_line*1000:.2f} ms/line, {lines_per_sec:.2f} lines/s)")

    return {
        "preds": preds,
        "CER": cer,
        "WER": wer,
        "total_seconds": total_time,
        "avg_seconds_per_line": avg_sec_per_line,
        "lines_per_second": lines_per_sec,
    }


# ---------------------------------------------------------------------
# Saving outputs (same pattern as zeroshot_donut.py)
# ---------------------------------------------------------------------

def save_outputs(
    out_dir: Path,
    ids: List[str],
    gts: List[str],
    preds: List[str],
    metrics: Dict[str, Any],
    model_name: str,
    manifest_path: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_txt = out_dir / "preds.txt"
    perline_csv = out_dir / "perline.csv"
    metrics_json = out_dir / "metrics.json"

    # preds.txt
    with preds_txt.open("w", encoding="utf-8", newline="") as f:
        for img_id, pred in zip(ids, preds):
            f.write(f"{img_id}\t{pred}\n")

    # perline.csv
    with perline_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_id", "gt", "pred", "CER"])
        for img_id, gt, pred in zip(ids, gts, preds):
            cer_line = cer_per_line(gt, pred)
            writer.writerow([img_id, gt, pred, f"{cer_line:.6f}"])

    # metrics.json
    meta: Dict[str, Any] = {
        "model": {
            "engine": "PARSeq",
            "hub_repo": "baudm/parseq",
            "variant": model_name,
        },
        "manifest": str(manifest_path),
        "num_lines": len(ids),
        "CER": metrics["CER"],
        "WER": metrics["WER"],
        "total_seconds": metrics["total_seconds"],
        "avg_seconds_per_line": metrics["avg_seconds_per_line"],
        "lines_per_second": metrics["lines_per_second"],
    }

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Outputs written to:")
    print("  ", preds_txt)
    print("  ", perline_csv)
    print("  ", metrics_json)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to manifest CSV")
    ap.add_argument(
        "--model",
        default="parseq",
        help="Model name for torch.hub baudm/parseq "
             "(e.g. 'parseq', 'parseq_tiny', 'abinet', 'crnn', 'trba', 'vitstr').",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (T4: 32 works fine for parseq).",
    )
    ap.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="Max decoding length (number of tokens).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of lines (for quick tests).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory; default: next to manifest, parseq_<model>.",
    )
    ap.add_argument(
        "--image_height",
        type=int,
        default=32,
        help="Input image height for PARSeq.",
    )
    ap.add_argument(
        "--image_width",
        type=int,
        default=128,
        help="Input image width for PARSeq.",
    )
    ap.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU even if GPU is available.",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 on GPU to save memory.",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    ids, img_paths, gts = load_manifest(str(manifest_path), limit=args.limit)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        variant = args.model.replace("/", "_")
        out_dir = manifest_path.parent / f"parseq_{variant}"

    metrics = run_parseq(
        model_name=args.model,
        ids=ids,
        img_paths=img_paths,
        gts=gts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        image_height=args.image_height,
        image_width=args.image_width,
        force_cpu=args.force_cpu,
        fp16=args.fp16,
    )

    save_outputs(
        out_dir=out_dir,
        ids=ids,
        gts=gts,
        preds=metrics["preds"],
        metrics=metrics,
        model_name=args.model,
        manifest_path=str(manifest_path),
    )


if __name__ == "__main__":
    main()
