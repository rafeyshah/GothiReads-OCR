#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Day 1 — CTC Alignment Report (FONT-SAFE + robust output selection)

Fixes the issue you hit (gt_len_match = 100% but clean_align = 0%):
- We DO NOT blindly decode the "first output" tensor anymore.
- We automatically pick the output tensor whose shape matches the class count (CTC logits/probs).
- We robustly convert that output to [T, C] before argmax over classes.

GT/font policy:
- GT units are Unicode codepoints (Python characters), with whitespace dropped (default).
  Combining marks count as separate units -> matches .font counting.

Inputs:
- manifest CSV with columns: image_path, txt_path, font_path, ok (optional), id (optional)

Outputs:
- alignment_report.json
- failures_top50.csv
- aligned_dataset.jsonl

Example:
python scripts/day1_ctc_alignment_report_font.py \
  --manifest /content/manifests/font/train_font_clean.csv \
  --rec_model_dir /content/PaddleOCR/inference/PP-OCRv5_server_rec \
  --rec_char_dict_path /content/PaddleOCR/ppocr/utils/gothi_dict.txt \
  --out_dir /content/runs/day1_font_alignment_codepoints \
  --use_gpu
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import regex as re


# ----------------------------
# GT + font helpers
# ----------------------------

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace").rstrip("\n")


def read_font_labels(p: Path) -> List[str]:
    raw = p.read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"\s+", "", raw)  # remove whitespace
    return list(raw)


def gt_units_for_font(text: str, drop_spaces: bool = True) -> List[str]:
    # Codepoints, not grapheme clusters. Combining marks count separately.
    units = list(text)
    if drop_spaces:
        units = [u for u in units if not u.isspace()]
    return units


# ----------------------------
# Paddle inference (rec model)
# ----------------------------

def load_char_dict(dict_path: Path) -> List[str]:
    """
    id2char[0] is blank.
    Remaining lines are characters/tokens.
    """
    lines = dict_path.read_text(
        encoding="utf-8", errors="replace").splitlines()
    chars = [ln.rstrip("\n") for ln in lines if ln.strip("\n") != ""]
    return ["<blank>"] + chars


def parse_inference_yml(rec_model_dir: Path) -> Dict:
    yml = rec_model_dir / "inference.yml"
    if not yml.exists():
        return {}
    import yaml
    return yaml.safe_load(yml.read_text(encoding="utf-8", errors="replace"))


def preprocess_rec_image(img_path: Path, rec_image_shape: str = "3,48,320") -> np.ndarray:
    """
    Minimal PaddleOCR-style preprocessing:
      - resize to target height, keep aspect ratio, pad to width
      - normalize to [-1, 1]
      - output shape [1, 3, H, W]
    """
    from PIL import Image
    c, h, w = [int(x) for x in rec_image_shape.split(",")]
    assert c == 3, "Expected 3-channel rec model."

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype("float32")

    src_h, src_w = img_np.shape[0], img_np.shape[1]
    ratio = h / float(src_h)
    new_w = int(np.round(src_w * ratio))
    new_w = max(1, min(new_w, w))

    img_resized = np.array(img.resize(
        (new_w, h), Image.BILINEAR)).astype("float32")

    padded = np.zeros((h, w, 3), dtype="float32")
    padded[:, :new_w, :] = img_resized

    padded = padded / 255.0
    padded = (padded - 0.5) / 0.5

    chw = np.transpose(padded, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def build_rec_predictor(rec_model_dir: Path, use_gpu: bool = True):
    import paddle.inference as paddle_infer

    model_file = rec_model_dir / "inference.pdmodel"
    params_file = rec_model_dir / "inference.pdiparams"
    if not model_file.exists() or not params_file.exists():
        raise FileNotFoundError(
            f"Missing inference.pdmodel/inference.pdiparams under {rec_model_dir}"
        )

    config = paddle_infer.Config(str(model_file), str(params_file))
    config.disable_glog_info()
    if use_gpu:
        config.enable_use_gpu(100, 0)
    else:
        config.disable_gpu()

    config.enable_memory_optim()
    try:
        config.switch_ir_optim(True)
    except Exception:
        pass

    return paddle_infer.create_predictor(config)


def run_rec_raw(predictor, input_name: str, output_name: str, x: np.ndarray) -> np.ndarray:
    input_handle = predictor.get_input_handle(input_name)
    input_handle.reshape(x.shape)
    input_handle.copy_from_cpu(x)

    predictor.run()

    out_handle = predictor.get_output_handle(output_name)
    return out_handle.copy_to_cpu()


def pick_best_output_name(predictor, num_classes: int) -> str:
    """
    Choose the output tensor most likely to be CTC logits by inspecting output shapes.

    We look for any output where a dimension is close to:
      - num_classes, or
      - num_classes - 1  (sometimes blank handling differs)
    Prefer 2D/3D tensors.
    """
    names = predictor.get_output_names()
    best_name = None
    best_score = 10**18

    for name in names:
        h = predictor.get_output_handle(name)
        shape = list(h.shape())

        # ignore empty/unknown
        if not shape:
            continue

        # score: closest dim to class count
        diffs = []
        for d in shape:
            if isinstance(d, int) and d > 0:
                diffs.append(min(abs(d - num_classes),
                             abs(d - (num_classes - 1))))
        if not diffs:
            continue
        score = min(diffs)

        # penalize weird dims (CTC logits are almost always 2D or 3D)
        if len(shape) not in (2, 3):
            score += 1000

        if score < best_score:
            best_score = score
            best_name = name

    return best_name if best_name is not None else names[0]


def to_time_class_matrix(raw: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert selected output to [T, C] for CTC decoding.

    Handles:
      - raw [B, T, C] or [B, C, T]
      - raw [T, C] or [C, T]

    Uses "closest to num_classes" to decide which axis is classes.
    """
    arr = np.array(raw)

    # squeeze batch if present
    if arr.ndim == 3:
        mat = arr[0]
    elif arr.ndim == 2:
        mat = arr
    else:
        raise RuntimeError(
            f"Unexpected selected output shape: {arr.shape}. Picked wrong output?")

    if mat.ndim != 2:
        raise RuntimeError(
            f"Expected 2D matrix after squeeze, got {mat.shape}")

    d0, d1 = mat.shape
    score0 = min(abs(d0 - num_classes), abs(d0 - (num_classes - 1)))
    score1 = min(abs(d1 - num_classes), abs(d1 - (num_classes - 1)))

    # If d1 looks like classes => mat is [T, C]
    if score1 <= score0:
        return mat

    # Else d0 looks like classes => mat is [C, T]
    return mat.T


# ----------------------------
# CTC best-path decode + spans
# ----------------------------

def ctc_best_path_with_spans(
    timestep_ids: List[int],
    id2char: List[str],
    blank_id: int = 0
) -> Tuple[str, List[Dict]]:
    """
    Collapse repeats, remove blanks, and keep spans.
    """
    aligned = []
    prev = None
    run_start = 0

    def flush(run_id: int, t0: int, t1: int):
        if run_id == blank_id:
            return
        ch = id2char[run_id] if 0 <= run_id < len(id2char) else "<UNK>"
        aligned.append({"ch": ch, "id": int(run_id),
                       "t0": int(t0), "t1": int(t1)})

    for t, cid in enumerate(timestep_ids):
        if prev is None:
            prev = cid
            run_start = t
            continue
        if cid != prev:
            flush(prev, run_start, t - 1)
            prev = cid
            run_start = t

    if prev is not None:
        flush(prev, run_start, len(timestep_ids) - 1)

    return "".join(a["ch"] for a in aligned), aligned


# ----------------------------
# Manifest helpers
# ----------------------------

def detect_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rec_model_dir", required=True)
    ap.add_argument("--rec_char_dict_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--drop_spaces", action="store_true", default=True)
    ap.add_argument("--blank_id", type=int, default=0)
    ap.add_argument("--output_tensor_name", default=None,
                    help="Optional: force a specific output tensor name.")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    rec_model_dir = Path(args.rec_model_dir)
    dict_path = Path(args.rec_char_dict_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # rec image shape from inference.yml (fallback to common default)
    cfg = parse_inference_yml(rec_model_dir)
    rec_image_shape = "3,48,320"
    if isinstance(cfg, dict) and "Global" in cfg and isinstance(cfg["Global"], dict):
        rec_image_shape = cfg["Global"].get("rec_image_shape", rec_image_shape)

    # id2char and class count
    id2char = load_char_dict(dict_path)
    num_classes = len(id2char)

    # predictor
    predictor = build_rec_predictor(rec_model_dir, use_gpu=bool(args.use_gpu))
    input_name = predictor.get_input_names()[0]
    output_names = predictor.get_output_names()

    # choose output tensor
    if args.output_tensor_name:
        chosen_out = args.output_tensor_name
    else:
        chosen_out = pick_best_output_name(predictor, num_classes)

    print("[INFO] Predictor outputs:", output_names)
    print("[INFO] Using output tensor:", chosen_out)
    print("[INFO] num_classes (dict+blank):",
          num_classes, "| blank_id:", args.blank_id)
    print("[INFO] rec_image_shape:", rec_image_shape)

    # load manifest rows
    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest}")

    cols = list(rows[0].keys())
    img_col = detect_col(cols, ["image_path", "img_path", "image", "img"])
    txt_col = detect_col(cols, ["txt_path"])
    font_col = detect_col(cols, ["font_path"])
    id_col = detect_col(cols, ["id", "img_id", "sample_id", "name"])
    ok_col = detect_col(cols, ["ok"])

    if img_col is None or txt_col is None or font_col is None:
        raise ValueError(f"Manifest missing required columns. Found: {cols}")

    total = 0
    gt_len_match = 0
    clean_align = 0
    failures = []

    aligned_jsonl_path = out_dir / "aligned_dataset.jsonl"
    with aligned_jsonl_path.open("w", encoding="utf-8") as jout:
        for r in rows:
            if args.limit is not None and total >= args.limit:
                break

            if ok_col is not None:
                if str(r.get(ok_col, "")).strip().upper() != "TRUE":
                    continue

            img_path = Path(r[img_col])
            txt_path = Path(r[txt_col])
            font_path = Path(r[font_col])

            if not img_path.exists() or not txt_path.exists() or not font_path.exists():
                continue

            sample_id = r.get(
                id_col, img_path.stem) if id_col else img_path.stem

            gt_text = read_text(txt_path)
            gt_units = gt_units_for_font(gt_text, drop_spaces=args.drop_spaces)
            gt_fonts = read_font_labels(font_path)

            ok_gt = (len(gt_units) == len(gt_fonts))
            if ok_gt:
                gt_len_match += 1

            # run model
            x = preprocess_rec_image(img_path, rec_image_shape=rec_image_shape)
            raw = run_rec_raw(predictor, input_name, chosen_out, x)

            # normalize to [T, C]
            out_tc = to_time_class_matrix(raw, num_classes=num_classes)

            # decode
            timestep_ids = out_tc.argmax(axis=1).tolist()
            aligned_text, aligned_chars = ctc_best_path_with_spans(
                timestep_ids, id2char, blank_id=args.blank_id
            )

            aligned_len = len(aligned_chars)
            gt_len = len(gt_units)

            ok_align = (aligned_len == gt_len)
            if ok_align:
                clean_align += 1
            else:
                failures.append({
                    "img_id": sample_id,
                    "image_path": str(img_path),
                    "gt_len": gt_len,
                    "aligned_len": aligned_len,
                    "gt_text": gt_text[:200],
                    "aligned_text": aligned_text[:200],
                })

            total += 1

            rec = {
                "id": sample_id,
                "image_path": str(img_path),
                "gt_text": gt_text,
                "gt_units": gt_units,
                "gt_fonts": gt_fonts,
                "aligned_text": aligned_text,
                "aligned_chars": aligned_chars,
                "ok_gt_len_match": ok_gt,
                "ok_align_len_match": ok_align,
                "rec_image_shape": rec_image_shape,
                "predictor_input": input_name,
                "predictor_output_used": chosen_out,
                "predictor_outputs_all": output_names,
                "num_classes": num_classes,
                "blank_id": args.blank_id,
            }
            jout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "manifest": str(manifest),
        "rec_model_dir": str(rec_model_dir),
        "rec_char_dict_path": str(dict_path),
        "total_evaluated": total,
        "gt_len_match_count": gt_len_match,
        "gt_len_match_pct": (gt_len_match / total * 100.0) if total else 0.0,
        "clean_align_count": clean_align,
        "clean_align_pct": (clean_align / total * 100.0) if total else 0.0,
        "criteria": "aligned_len == gt_units_len (codepoints, drop whitespace)",
        "chosen_output_tensor": chosen_out,
        "notes": [
            "GT units are codepoints (characters), not grapheme clusters.",
            "Combining marks count separately to match .font and CTC behavior.",
            "We auto-select the CTC output tensor by matching class dimension to dict size.",
        ],
    }

    (out_dir / "alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    failures_sorted = sorted(
        failures, key=lambda x: abs(x["aligned_len"] - x["gt_len"]), reverse=True
    )
    top50 = failures_sorted[:50]
    with (out_dir / "failures_top50.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_id", "image_path", "gt_len",
                   "aligned_len", "gt_text", "aligned_text"])
        for x in top50:
            w.writerow([x["img_id"], x["image_path"], x["gt_len"],
                       x["aligned_len"], x["gt_text"], x["aligned_text"]])

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["clean_align_pct"] < 95.0:
        print(
            f"\n[WARN] clean_align_pct={report['clean_align_pct']:.2f}% < 95%. Check failures_top50.csv\n")


if __name__ == "__main__":
    main()
