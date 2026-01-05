"""
Day 3 unified evaluation harness for Gothi-Read (ICDAR 2024 Track B)
Usage:
  python scripts/day3_harness.py --manifest manifests/valid.csv --model microsoft/trocr-base-printed --height 64 --limit 500
"""

import os
import csv
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image, ImageOps

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
except Exception:
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None
    torch = None

try:
    from jiwer import cer as jiwer_cer, wer as jiwer_wer
except Exception:
    jiwer_cer = None
    jiwer_wer = None


def _edit_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, m + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def cer_per_line(ref: str, hyp: str) -> float:
    return _edit_distance(ref, hyp) / max(1, len(ref))


def preprocess_images(paths: List[str], target_height: int) -> List[Image.Image]:
    resized, widths = [], []
    for p in paths:
        img = Image.open(p).convert("L")
        w, h = img.size or (1, 1)
        scale = target_height / float(h)
        new_w = max(1, int(round(w * scale)))
        resized_img = img.resize((new_w, target_height), Image.BICUBIC)
        resized.append(resized_img)
        widths.append(new_w)
    max_w = max(widths) if widths else 1
    out = []
    for img in resized:
        pad_w = max_w - img.size[0]
        padded = ImageOps.expand(img, border=(0, 0, pad_w, 0), fill=255)
        out.append(padded.convert("RGB"))
    return out


def predict_lines(model_name: str, images: List[Image.Image], max_length: int = 256, **generate_kwargs) -> List[str]:
    if TrOCRProcessor is None or VisionEncoderDecoderModel is None:
        raise RuntimeError("Transformers not available.")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()
    device = torch.device(
        "cuda" if torch and torch.cuda.is_available() else "cpu")
    model.to(device)
    preds = []
    batch_size = 8
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_length=max_length, **generate_kwargs)
        texts = processor.batch_decode(out_ids, skip_special_tokens=True)
        preds.extend(texts)
    return preds


def evaluate(preds: List[str], refs: List[str], ids: List[str]) -> Dict:
    if jiwer_cer and jiwer_wer:
        overall_cer = float(jiwer_cer(refs, preds))
        overall_wer = float(jiwer_wer(refs, preds))
    else:
        total_edits = sum(_edit_distance(r, p) for r, p in zip(refs, preds))
        total_chars = max(1, sum(len(r) for r in refs))
        overall_cer = total_edits / total_chars
        overall_wer = None

    per_line = [dict(img_id=i, gt=g, pred=p, CER=cer_per_line(g, p))
                for i, g, p in zip(ids, refs, preds)]

    def infer_fontmix(
        i): return "single" if "single" in i else "multiple" if "multiple" in i else "unknown"
    def infer_book(i): parts = i.split(
        "/"); return parts[-2] if len(parts) >= 2 else "unknown"

    per_book, per_fontmix = {}, {}
    for row in per_line:
        b, fm = infer_book(row["img_id"]), infer_fontmix(row["img_id"])
        for group, store in [(b, per_book), (fm, per_fontmix)]:
            d = store.setdefault(group, {"edits": 0, "chars": 0, "lines": 0})
            d["edits"] += _edit_distance(row["gt"], row["pred"])
            d["chars"] += max(1, len(row["gt"]))
            d["lines"] += 1
    for store in (per_book, per_fontmix):
        for k, v in store.items():
            v["CER"] = v["edits"] / v["chars"]
    return {"CER": overall_cer, "WER": overall_wer, "per_book": per_book, "per_fontmix": per_fontmix}


def load_val_split(csv_path: str, limit=None) -> Tuple[List[str], List[str], List[str]]:
    ids, imgs, gts = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # accept TRUE/True/1 (keep backwards compatibility)
            ok = str(r.get("ok", "")).strip().upper()
            if ok not in {"TRUE", "1", "YES"}:
                continue

            img = (r.get("image_path") or "").strip()
            if not img:
                continue

            # NEW: prefer gt_text if present
            gt_text = r.get("gt_text")
            if gt_text is not None and str(gt_text).strip() != "":
                gt = str(gt_text)
            else:
                # OLD: fall back to txt_path
                txt = (r.get("txt_path") or "").strip()
                if not txt:
                    continue
                try:
                    gt = Path(txt).read_text(encoding="utf-8")
                except Exception:
                    continue

            # Keep exact behavior: strip only trailing newlines, not internal spaces
            gt = gt.strip()

            ids.append(r.get("id", Path(img).stem))
            imgs.append(img)
            gts.append(gt)

            if limit and len(ids) >= limit:
                break
    return ids, imgs, gts


def run_harness(manifest: str, model: str, height: int, limit=None, max_length=256, num_beams=1, length_penalty=1.0, no_repeat_ngram_size=0, early_stopping=True):
    ids, imgs, gts = load_val_split(manifest, limit)
    images = preprocess_images(imgs, height)
    generate_kwargs = dict(
        num_beams=num_beams,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping,
    )

    preds = predict_lines(model, images, max_length, **generate_kwargs)
    metrics = evaluate(preds, gts, ids)
    metrics["model"] = model
    metrics["decode"] = {
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "early_stopping": bool(early_stopping),
    }
    # so every run documents the split
    metrics["val_manifest"] = str(Path(args.manifest))
    out_dir = Path("runs") / model.replace("/", "_") / \
        datetime.now().strftime("%Y%m%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"preds.txt", "w", encoding="utf-8") as f:
        for i, p in zip(ids, preds):
            f.write(f"{i}\t{p}\n")
    with open(out_dir/"per_line.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_id", "gt", "pred", "CER"])
        for i, g, p in zip(ids, gts, preds):
            w.writerow([i, g, p, cer_per_line(g, p)])
    with open(out_dir/"metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Saved results to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="microsoft/trocr-base-printed")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=1,
                    help="Beam size (1 = greedy)")
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--early_stopping", action="store_true")
    args = ap.parse_args()
    run_harness(args.manifest, args.model, args.height, args.limit, args.max_length,
                args.num_beams, args.length_penalty, args.no_repeat_ngram_size, bool(args.early_stopping))
