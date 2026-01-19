#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import cv2  # type: ignore
import paddle


class AlignJsonlFontDataset(paddle.io.Dataset):
    """
    Dataset backed by extract_align.py output JSONL.

    Each item returns:
      - x: float32 [C,H,W] preprocessed image (PP-OCR style)
      - ranges: int64 [G,2] inclusive timestep ranges
      - y: int64 [G] font class ids

    NOTE: pooling is done in the training step (needs encoder features).
    """

    def __init__(
        self,
        align_jsonl: Path,
        font2id: Dict[str, int],
        rec_image_shape: Tuple[int, int, int] = (3, 32, 320),
        max_graphemes: int = 80,
    ):
        self.align_jsonl = Path(align_jsonl)
        self.font2id = font2id
        self.rec_image_shape = rec_image_shape
        self.max_graphemes = max_graphemes

        self.items: List[Dict[str, Any]] = []
        with self.align_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if not ex.get("ok_align"):
                    continue
                img = ex.get("image_path")
                t_ranges = ex.get("t_ranges")
                gt_fonts = ex.get("gt_fonts")
                if not img or not isinstance(t_ranges, list) or not isinstance(gt_fonts, list):
                    continue
                if len(t_ranges) != len(gt_fonts):
                    continue
                if len(gt_fonts) == 0 or len(gt_fonts) > self.max_graphemes:
                    continue

                # Map labels; drop sample if unknown label encountered
                y = []
                ok = True
                for lab in gt_fonts:
                    lab = str(lab)
                    if lab not in self.font2id:
                        ok = False
                        break
                    y.append(self.font2id[lab])
                if not ok:
                    continue

                self.items.append({
                    "image_path": str(img),
                    "t_ranges": t_ranges,
                    "y": y,
                })

    def __len__(self) -> int:
        return len(self.items)

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        # Same preprocessing style as your extractor (resize H, pad W, normalize) :contentReference[oaicite:4]{index=4}
        c, H, W = self.rec_image_shape
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ratio = w / float(h)
        target_w = min(W, int(np.ceil(H * ratio)))
        target_w = max(1, target_w)
        resized = cv2.resize(img, (target_w, H),
                             interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((H, W, 3), dtype=np.uint8)
        padded[:, :target_w, :] = resized

        x = padded.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (2, 0, 1))  # CHW
        return x

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = cv2.imread(it["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {it['image_path']}")

        x = self._preprocess(img)  # [3,H,W]
        ranges = np.array(it["t_ranges"], dtype=np.int64)  # [G,2]
        y = np.array(it["y"], dtype=np.int64)  # [G]
        return x, ranges, y


def collate_font_batch(batch):
    """
    Collate variable-G samples.
    We pad ranges/y to max_G and provide a mask.
    """
    xs, ranges_list, ys_list = zip(*batch)
    xs = np.stack(xs, axis=0).astype("float32")  # [B,3,H,W]

    maxG = max(r.shape[0] for r in ranges_list)
    B = len(batch)

    ranges_pad = np.zeros((B, maxG, 2), dtype="int64")
    y_pad = np.zeros((B, maxG), dtype="int64")
    mask = np.zeros((B, maxG), dtype="float32")

    for i, (r, y) in enumerate(zip(ranges_list, ys_list)):
        g = r.shape[0]
        ranges_pad[i, :g, :] = r
        y_pad[i, :g] = y
        mask[i, :g] = 1.0

    return paddle.to_tensor(xs), paddle.to_tensor(ranges_pad), paddle.to_tensor(y_pad), paddle.to_tensor(mask)
