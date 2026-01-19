#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_align_dp_t4.py
Phase-2 alignment for ICDAR2024 Track-B (multi-font group recognition)

Key points:
- Uses your canonical units.py policy (NFC + grapheme clusters; keep spaces, drop layout ws)  ✅
- Runs PaddleOCR recognition inference (exported inference.pdmodel/.pdiparams)
- Best-path CTC decode + timestep ranges
- Converts decoded tokens -> pred_units (graphemes)
- DP align pred_units ↔ gt_units (Levenshtein backtrace)
- Builds GT-length t_ranges by mapping each GT grapheme to a predicted grapheme's [t0,t1]
- Applies professional gating (Tier-A defaults)

Output JSONL:
  ok_align True rows contain:
    id, image_path, gt_units, gt_fonts, t_ranges (len==len(gt_units)),
    pred_text, pred_units, T, C, align stats
  ok_align False rows contain:
    id, image_path, reason, + debug stats
"""

from __future__ import annotations

from units import normalize, strip_ws, graphemes, font_tokens  # your uploaded units.py

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unicodedata
import regex as re
import numpy as np
import cv2  # type: ignore
import yaml

import paddle.inference as paddle_infer  # type: ignore

DICT_WARN_ONCE = False
EXTEND_INFO_ONCE = False
_GRAPHEME_RE = re.compile(r"\X", re.U)


# ----------------------------
# Dict handling (same strategy as your previous file)
# ----------------------------

def read_dict_tokens_keep_spaces(path: Path) -> List[str]:
    toks: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            toks.append(line.rstrip("\n"))
    if toks and toks[-1] == "":
        toks = toks[:-1]

    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def read_dict_tokens_from_infer_cfg(rec_model_dir: Path) -> List[str]:
    rec_model_dir = Path(rec_model_dir)
    for name in [
        "infer_cfg.yml", "infer_cfg.yaml",
        "inference.yml", "inference.yaml",
        "inference (1).yml", "inference (1).yaml",
    ]:
        p = rec_model_dir / name
        if p.exists():
            y = yaml.safe_load(p.read_text(encoding="utf-8"))
            pp = (y or {}).get("PostProcess", {}
                               ) if isinstance(y, dict) else {}
            char_list = pp.get("character_dict")
            use_space = bool(pp.get("use_space_char", False))
            if isinstance(char_list, list) and len(char_list) > 0:
                toks = ["" if c is None else str(c) for c in char_list]
                if use_space and " " not in toks:
                    toks.append(" ")
                return toks

            dict_path = pp.get("character_dict_path") or (
                y or {}).get("character_dict_path")
            if dict_path:
                dp = Path(dict_path)
                if not dp.is_absolute():
                    dp = rec_model_dir / dp
                if dp.exists():
                    return read_dict_tokens_keep_spaces(dp)
            break

    raise FileNotFoundError(
        f"No usable infer_cfg.yml/inference.yml with PostProcess.character_dict in {rec_model_dir}")


def build_id2tok_for_ctc(dict_tokens: List[str], C: int) -> Tuple[List[str], int, int]:
    """
    PaddleOCR CTC expectation:
      model C == (len(dict_tokens) + 1 blank) -> dict_len == C-1

    Handles drift:
      - dict_len == C-1 : OK
      - dict_len == C   : dict already includes blank at index 0 (rare)
      - dict_len < C-1  : model has extra tail classes -> slice logits to dict_len+1
    """
    dict_len = len(dict_tokens)

    if dict_len == C - 1:
        id2tok = ["<BLANK>"] + dict_tokens
        return id2tok, 0, len(id2tok)

    if dict_len == C:
        return dict_tokens, 0, dict_len

    if dict_len < C - 1:
        id2tok = ["<BLANK>"] + dict_tokens
        C_used = len(id2tok)
        global DICT_WARN_ONCE
        if not DICT_WARN_ONCE:
            print(
                f"[DICT][WARN] dict_len={dict_len} < C-1={C-1}. Slicing logits C={C} -> C_used={C_used}")
            DICT_WARN_ONCE = True
        return id2tok, 0, C_used

    raise RuntimeError(
        f"DICT_MISMATCH: dict_len={dict_len}, model_C={C}. Dict longer than logits.")


def is_combining_token(tok: str) -> bool:
    """
    True if token is purely combining marks / modifier marks.
    Handles single-char marks like '́' and multi-mark strings.
    """
    if tok == "":
        return False
    # If every codepoint is a combining mark (category Mn/Mc/Me) or combining() > 0
    for ch in tok:
        if unicodedata.combining(ch) == 0 and unicodedata.category(ch) not in ("Mn", "Mc", "Me"):
            return False
    return True


def build_units_and_ranges_from_tokens(
    pred_tokens: Sequence[str],
    token_ranges: Sequence[Sequence[int]],
) -> Tuple[List[str], List[List[int]], str]:
    """
    Build pred_units and pred_g_ranges aligned 1:1.

    Key behavior:
    - keep ordinary spaces (your units.py policy keeps spaces) :contentReference[oaicite:1]{index=1}
    - merge combining-mark tokens into previous unit and extend its time range
    - also split multi-grapheme tokens (rare) into separate units (same time range)
    """
    units: List[str] = []
    ranges: List[List[int]] = []
    text_parts: List[str] = []

    for tok, r in zip(pred_tokens, token_ranges):
        # apply same normalization/whitespace policy used in units.py
        tok2 = strip_ws(normalize(tok))
        if tok2 == "":
            continue

        t0, t1 = int(r[0]), int(r[1])
        text_parts.append(tok2)

        if is_combining_token(tok2) and units:
            # merge into previous grapheme cluster
            units[-1] = units[-1] + tok2
            ranges[-1][1] = max(ranges[-1][1], t1)
            continue

        # split token into grapheme clusters (within token)
        gs = _GRAPHEME_RE.findall(tok2)
        if len(gs) <= 1:
            units.append(tok2)
            ranges.append([t0, t1])
        else:
            # token contains multiple graphemes; assign same [t0,t1] to each
            for g in gs:
                units.append(g)
                ranges.append([t0, t1])

    pred_text_clean = "".join(text_parts)
    return units, ranges, pred_text_clean
# ----------------------------
# CTC best-path + ranges
# ----------------------------


def best_path_ctc_with_ranges(logits_T_C: np.ndarray, blank_id: int) -> Tuple[List[int], List[List[int]]]:
    ids = np.argmax(logits_T_C, axis=1).astype(np.int32)  # [T]

    sym_ids: List[int] = []
    sym_ranges: List[List[int]] = []
    prev: Optional[int] = None
    run_start: Optional[int] = None

    for t, idx in enumerate(ids.tolist()):
        if idx == blank_id:
            if prev is not None and run_start is not None:
                sym_ids.append(prev)
                sym_ranges.append([run_start, t - 1])
            prev = None
            run_start = None
            continue

        if prev is None:
            prev = idx
            run_start = t
        elif idx == prev:
            pass
        else:
            if run_start is not None:
                sym_ids.append(prev)
                sym_ranges.append([run_start, t - 1])
            prev = idx
            run_start = t

    if prev is not None and run_start is not None:
        sym_ids.append(prev)
        sym_ranges.append([run_start, len(ids) - 1])

    return sym_ids, sym_ranges


def ids_to_tokens(sym_ids: Sequence[int], id2tok: Sequence[str]) -> List[str]:
    return [id2tok[i] if 0 <= i < len(id2tok) else "" for i in sym_ids]


def merge_token_ranges_to_grapheme_ranges(
    pred_tokens: Sequence[str],
    token_ranges: Sequence[Sequence[int]],
    pred_graphemes: Sequence[str],
) -> Optional[List[List[int]]]:
    if len(pred_tokens) != len(token_ranges):
        return None

    g_ranges: List[List[int]] = []
    j = 0
    for g in pred_graphemes:
        if j >= len(pred_tokens):
            return None

        acc = ""
        t0: Optional[int] = None
        t1: Optional[int] = None
        while j < len(pred_tokens) and len(acc) < len(g):
            acc += pred_tokens[j]
            r = token_ranges[j]
            if t0 is None:
                t0 = int(r[0])
            t1 = int(r[1])
            j += 1
            if acc == g:
                break

        if acc != g or t0 is None or t1 is None:
            return None
        g_ranges.append([t0, t1])

    if j != len(pred_tokens):
        return None

    return g_ranges


def expand_ranges_to_graphemes_keep_spaces(
    pred_tokens: Sequence[str],
    token_ranges: Sequence[Sequence[int]],
) -> List[List[int]]:
    """
    Expand token-level ranges to grapheme-level ranges.

    IMPORTANT for your units.py policy:
    - Keep ordinary spaces as units -> DO NOT drop whitespace graphemes.
    - Each grapheme produced from a token gets the token's [t0,t1].
    """
    out: List[List[int]] = []
    for tok, r in zip(pred_tokens, token_ranges):
        gs = graphemes(tok)  # uses your policy: keeps spaces, drops layout ws
        for _g in gs:
            out.append([int(r[0]), int(r[1])])
    return out


# ----------------------------
# DP alignment (Levenshtein backtrace)
# ----------------------------

def dp_align_units(pred_units: List[str], gt_units: List[str]) -> Tuple[List[Tuple[Optional[int], Optional[int], str]], Dict[str, Any]]:
    n, m = len(pred_units), len(gt_units)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up(I), 2 left(D)

    for i in range(1, n + 1):
        dp[i, 0] = i
        bt[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = j
        bt[0, j] = 2

    for i in range(1, n + 1):
        pu = pred_units[i - 1]
        for j in range(1, m + 1):
            gu = gt_units[j - 1]
            sub_cost = 0 if pu == gu else 1

            diag = dp[i - 1, j - 1] + sub_cost
            up = dp[i - 1, j] + 1
            left = dp[i, j - 1] + 1

            best = diag
            code = 0
            if up < best:
                best = up
                code = 1
            if left < best:
                best = left
                code = 2

            dp[i, j] = best
            bt[i, j] = code

    i, j = n, m
    rev: List[Tuple[Optional[int], Optional[int], str]] = []
    matches = subs = pred_gaps = gt_gaps = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i, j] == 0:
            pi, gj = i - 1, j - 1
            if pred_units[pi] == gt_units[gj]:
                rev.append((pi, gj, "M"))
                matches += 1
            else:
                rev.append((pi, gj, "S"))
                subs += 1
            i -= 1
            j -= 1
        elif i > 0 and bt[i, j] == 1:
            rev.append((i - 1, None, "I"))
            pred_gaps += 1
            i -= 1
        else:
            rev.append((None, j - 1, "D"))
            gt_gaps += 1
            j -= 1

    path = list(reversed(rev))
    edits = int(dp[n, m])

    max_gt_gap_run = 0
    cur = 0
    for _, gj, op in path:
        if op == "D" and gj is not None:
            cur += 1
            max_gt_gap_run = max(max_gt_gap_run, cur)
        else:
            cur = 0

    covered_gt = matches + subs
    coverage = (covered_gt / m) if m > 0 else 0.0

    stats = {
        "edits": edits,
        "matches": matches,
        "subs": subs,
        "gt_gaps": gt_gaps,
        "pred_gaps": pred_gaps,
        "gt_len": m,
        "pred_len": n,
        "max_gt_gap_run": max_gt_gap_run,
        "coverage": float(coverage),
    }
    return path, stats


def build_gt_t_ranges_from_alignment(
    align_path: List[Tuple[Optional[int], Optional[int], str]],
    pred_g_ranges: List[List[int]],
    gt_len: int,
) -> Tuple[List[List[int]], int]:
    gt_to_pred: List[Optional[int]] = [None] * gt_len
    for pi, gj, op in align_path:
        if gj is None:
            continue
        if op in ("M", "S"):
            gt_to_pred[gj] = pi

    left_pred: List[Optional[int]] = [None] * gt_len
    last: Optional[int] = None
    for j in range(gt_len):
        if gt_to_pred[j] is not None:
            last = gt_to_pred[j]
        left_pred[j] = last

    right_pred: List[Optional[int]] = [None] * gt_len
    last = None
    for j in reversed(range(gt_len)):
        if gt_to_pred[j] is not None:
            last = gt_to_pred[j]
        right_pred[j] = last

    out: List[List[int]] = []
    filled = 0
    for j in range(gt_len):
        pi = gt_to_pred[j]
        if pi is not None and 0 <= pi < len(pred_g_ranges):
            out.append([int(pred_g_ranges[pi][0]), int(pred_g_ranges[pi][1])])
        else:
            pick = left_pred[j] if left_pred[j] is not None else right_pred[j]
            if pick is None:
                out.append([0, 0])
            else:
                out.append([int(pred_g_ranges[pick][0]),
                           int(pred_g_ranges[pick][1])])
            filled += 1
    return out, filled


# ----------------------------
# Image preprocessing (PP-OCR style)
# ----------------------------

def preprocess_rec_image(img_bgr: np.ndarray, rec_shape: Tuple[int, int, int]) -> np.ndarray:
    c, H, W = rec_shape
    if c != 3:
        raise ValueError(f"Expected C=3, got {rec_shape}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    ratio = w / float(h)
    target_w = min(W, int(math.ceil(H * ratio)))
    target_w = max(1, target_w)

    resized = cv2.resize(img, (target_w, H), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((H, W, 3), dtype=np.uint8)
    padded[:, :target_w, :] = resized

    x = padded.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))  # CHW
    return x


# ----------------------------
# Paddle Inference wrapper
# ----------------------------

class RecPredictor:
    def __init__(self, model_dir: Path, use_gpu: bool, gpu_id: int, gpu_mem_mb: int, cpu_threads: int):
        model_path = model_dir / "inference.pdmodel"
        params_path = model_dir / "inference.pdiparams"
        if not model_path.exists() or not params_path.exists():
            raise FileNotFoundError(
                f"Missing inference.pdmodel / inference.pdiparams in {model_dir}")

        cfg = paddle_infer.Config(str(model_path), str(params_path))
        if use_gpu:
            cfg.enable_use_gpu(gpu_mem_mb, gpu_id)
        else:
            cfg.disable_gpu()
            cfg.set_cpu_math_library_num_threads(cpu_threads)

        cfg.switch_use_feed_fetch_ops(False)
        cfg.switch_ir_optim(True)

        self.predictor = paddle_infer.create_predictor(cfg)
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()

    def run_logits(self, x_bchw: np.ndarray) -> np.ndarray:
        inp = self.predictor.get_input_handle(self.input_names[0])
        inp.reshape(x_bchw.shape)
        inp.copy_from_cpu(x_bchw)
        self.predictor.run()

        for name in self.output_names:
            arr = self.predictor.get_output_handle(name).copy_to_cpu()
            if arr.ndim == 3:
                return arr.astype(np.float32)
        raise RuntimeError(f"No 3D output found. outputs={self.output_names}")


def normalize_logits_to_BTC(logits: np.ndarray, dict_len: int) -> Tuple[np.ndarray, int, int]:
    if logits.ndim != 3:
        raise RuntimeError(f"Unexpected logits shape {logits.shape}")

    B, D1, D2 = logits.shape
    c_candidates = {dict_len - 2, dict_len - 1,
                    dict_len, dict_len + 1, dict_len + 2}

    if D2 in c_candidates:
        return logits, D1, D2
    if D1 in c_candidates:
        return np.transpose(logits, (0, 2, 1)), D2, D1
    if D2 > D1:
        return logits, D1, D2

    raise RuntimeError(
        f"Cannot locate vocab axis. logits={logits.shape}, dict_len={dict_len}")


# ----------------------------
# IO helpers
# ----------------------------

def load_rows_ok_true(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [row for row in r if row.get("ok") == "TRUE"]


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--rec_model_dir", type=Path, required=True)
    ap.add_argument("--rec_char_dict_path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--gpu_mem_mb", type=int, default=4000)
    ap.add_argument("--cpu_threads", type=int, default=4)
    ap.add_argument("--rec_image_shape", type=str, default="3,32,320")

    # Tier-A defaults (pro)
    ap.add_argument("--min_coverage", type=float, default=0.90)
    ap.add_argument("--max_gt_gap_run", type=int, default=1)
    ap.add_argument("--max_edits_abs", type=int, default=2)
    ap.add_argument("--max_edits_frac", type=float, default=0.10)

    ap.add_argument("--debug_rejects", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=5000)

    args = ap.parse_args()

    rec_shape = tuple(int(x.strip()) for x in args.rec_image_shape.split(","))
    if len(rec_shape) != 3:
        raise SystemExit("--rec_image_shape must be like 3,32,320")

    # Load dictionary tokens (same as your previous file)
    try:
        dict_tokens = read_dict_tokens_from_infer_cfg(Path(args.rec_model_dir))
        print(
            f"[DICT] loaded {len(dict_tokens)} tokens from infer_cfg in {args.rec_model_dir}")
    except Exception as e:
        dict_tokens = read_dict_tokens_keep_spaces(
            Path(args.rec_char_dict_path))
        print(
            f"[DICT][WARN] falling back to rec_char_dict_path={args.rec_char_dict_path} ({e})")

    predictor = RecPredictor(
        model_dir=args.rec_model_dir,
        use_gpu=bool(args.use_gpu),
        gpu_id=args.gpu_id,
        gpu_mem_mb=args.gpu_mem_mb,
        cpu_threads=args.cpu_threads,
    )

    rows = load_rows_ok_true(args.manifest)
    if args.max_rows and len(rows) > args.max_rows:
        rows = rows[: args.max_rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    printed = 0

    fail: Dict[str, int] = {
        "LOAD_ERROR": 0,
        "INFER_ERROR": 0,
        "LOGITS_SHAPE_ERROR": 0,
        "DICT_MISMATCH": 0,
        "GT_LEN_MISMATCH": 0,
        "RANGE_MAPPING_FAILED": 0,
        "DP_REJECT": 0,
    }

    with args.out.open("w", encoding="utf-8") as out_f:
        for i0 in range(0, len(rows), args.batch_size):
            batch = rows[i0:i0 + args.batch_size]

            xs: List[np.ndarray] = []
            metas: List[Dict[str, Any]] = []

            for r in batch:
                total += 1
                sid = r.get("id", "")
                try:
                    img = cv2.imread(r["image_path"], cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("cv2.imread returned None")

                    xs.append(preprocess_rec_image(img, rec_shape))

                    gt_raw = read_text(Path(r["txt_path"]))
                    font_raw = read_text(Path(r["font_path"]))

                    gt_units = graphemes(gt_raw)
                    gt_fonts = font_tokens(
                        font_raw, expected_len=len(gt_units))

                    metas.append({
                        "id": sid,
                        "image_path": r["image_path"],
                        "gt_units": gt_units,
                        "gt_fonts": gt_fonts,
                    })
                except Exception:
                    fail["LOAD_ERROR"] += 1
                    out_f.write(json.dumps({
                        "id": sid,
                        "image_path": r.get("image_path", ""),
                        "ok_align": False,
                        "reason": "LOAD_ERROR",
                    }, ensure_ascii=False) + "\n")

            if not xs:
                continue

            x_bchw = np.stack(xs, axis=0).astype(np.float32)

            try:
                raw_logits = predictor.run_logits(x_bchw)
            except Exception as e:
                fail["INFER_ERROR"] += len(metas)
                for m in metas:
                    out_f.write(json.dumps({
                        "id": m["id"],
                        "image_path": m["image_path"],
                        "ok_align": False,
                        "reason": "INFER_ERROR",
                        "error": repr(e),
                    }, ensure_ascii=False) + "\n")
                continue

            try:
                logits_btc, T, C = normalize_logits_to_BTC(
                    raw_logits, dict_len=len(dict_tokens))
            except Exception as e:
                fail["LOGITS_SHAPE_ERROR"] += len(metas)
                for m in metas:
                    out_f.write(json.dumps({
                        "id": m["id"],
                        "image_path": m["image_path"],
                        "ok_align": False,
                        "reason": "LOGITS_SHAPE_ERROR",
                        "raw_logits_shape": list(raw_logits.shape),
                        "error": repr(e),
                    }, ensure_ascii=False) + "\n")
                continue

            # SPACE patch (same idea as your previous file)
            dict_tokens_eff = dict_tokens
            if len(dict_tokens_eff) == C - 2:
                dict_tokens_eff = dict_tokens_eff + [" "]
                global EXTEND_INFO_ONCE
                if not EXTEND_INFO_ONCE:
                    print(
                        f"[DICT][INFO] auto-extended dict by 1 token: {len(dict_tokens)} -> {len(dict_tokens_eff)} (added SPACE)")
                    EXTEND_INFO_ONCE = True

            try:
                id2tok, blank_id, C_used = build_id2tok_for_ctc(
                    dict_tokens_eff, C)
                if C_used != C:
                    logits_btc = logits_btc[:, :, :C_used]
                    C = C_used
            except Exception as e:
                fail["DICT_MISMATCH"] += len(metas)
                for m in metas:
                    out_f.write(json.dumps({
                        "id": m["id"],
                        "image_path": m["image_path"],
                        "ok_align": False,
                        "reason": "DICT_MISMATCH",
                        "C": int(C),
                        "dict_len": int(len(dict_tokens_eff)),
                        "error": repr(e),
                    }, ensure_ascii=False) + "\n")
                continue

            B = logits_btc.shape[0]
            for b in range(min(B, len(metas))):
                m = metas[b]
                gt_units: List[str] = m["gt_units"]
                gt_fonts: List[str] = m["gt_fonts"]

                if len(gt_units) != len(gt_fonts):
                    fail["GT_LEN_MISMATCH"] += 1
                    out_f.write(json.dumps({
                        "id": m["id"],
                        "image_path": m["image_path"],
                        "ok_align": False,
                        "reason": "GT_LEN_MISMATCH",
                        "gt_units_len": len(gt_units),
                        "gt_fonts_len": len(gt_fonts),
                    }, ensure_ascii=False) + "\n")
                    continue

                sym_ids, sym_ranges = best_path_ctc_with_ranges(
                    logits_btc[b], blank_id=blank_id)
                pred_tokens = ids_to_tokens(sym_ids, id2tok)
                
                pred_units, g_ranges, pred_clean = build_units_and_ranges_from_tokens(pred_tokens, sym_ranges)

                # If you want to be extra safe, you can verify:
                # if pred_units != graphemes(pred_clean):
                #     (not required; combining merges may make string-level graphemes match anyway)


                align_path, st = dp_align_units(pred_units, gt_units)
                edit_budget = max(int(args.max_edits_abs), int(
                    math.ceil(args.max_edits_frac * max(1, st["gt_len"]))))

                ok = (
                    st["coverage"] >= float(args.min_coverage)
                    and st["max_gt_gap_run"] <= int(args.max_gt_gap_run)
                    and st["edits"] <= edit_budget
                )

                if not ok:
                    fail["DP_REJECT"] += 1
                    if args.debug_rejects and printed < args.debug_rejects:
                        printed += 1
                        print("---- DP reject ----")
                        print("id:", m["id"])
                        print("gt_len:", st["gt_len"],
                              "pred_len:", st["pred_len"])
                        print("edits:", st["edits"], "budget:", edit_budget,
                              "coverage:", f"{st['coverage']:.3f}",
                              "max_gt_gap_run:", st["max_gt_gap_run"])
                        print("pred_clean[:160]:", pred_clean[:160])
                        print("------------------")

                    out_f.write(json.dumps({
                        "id": m["id"],
                        "image_path": m["image_path"],
                        "ok_align": False,
                        "reason": "DP_REJECT",
                        "pred_text": pred_clean,
                        "align": st,
                        "edit_budget": int(edit_budget),
                    }, ensure_ascii=False) + "\n")
                    continue

                gt_t_ranges, filled = build_gt_t_ranges_from_alignment(
                    align_path=align_path,
                    pred_g_ranges=g_ranges,
                    gt_len=len(gt_units),
                )

                kept += 1
                out_f.write(json.dumps({
                    "id": m["id"],
                    "image_path": m["image_path"],
                    "gt_units": gt_units,
                    "gt_fonts": gt_fonts,
                    "t_ranges": gt_t_ranges,    # GT-length
                    "ok_align": True,
                    "pred_text": pred_clean,
                    "pred_units": pred_units,
                    "T": int(T),
                    "C": int(C),
                    "align": st,
                    "filled_gt_gaps": int(filled),
                }, ensure_ascii=False) + "\n")

            if args.progress_every and args.progress_every > 0 and total % args.progress_every == 0:
                pct = (kept / total * 100.0) if total else 0.0
                top3 = sorted(fail.items(), key=lambda kv: -kv[1])[:3]
                print(
                    f"[PROGRESS] rows={total} kept={kept} ({pct:.2f}%) fails={{{', '.join([f'{k}:{v}' for k, v in top3])}}}")

    pct = (kept / total * 100.0) if total else 0.0
    print(f"[DONE] wrote: {args.out}")
    print(
        f"[STATS] kept_aligned={kept} / total_rows_seen={total} ({pct:.2f}%)")
    print("[FAIL_REASONS]")
    for k, v in sorted(fail.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
