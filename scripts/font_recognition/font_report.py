#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import paddle
import paddle.nn as nn

from font_dataset import AlignJsonlFontDataset, collate_font_batch
from rec_loader import load_rec_model_with_features


# -----------------------
# Pooling per grapheme
# -----------------------
def pool_by_ranges(F_btd: paddle.Tensor, ranges_bG2: paddle.Tensor, pooling: str) -> paddle.Tensor:
    """
    F_btd: [B,T,D]
    ranges: [B,G,2] inclusive
    pooling: 'mean' | 'max' | 'meanmax'
    returns:
      - mean/max:    [B,G,D]
      - meanmax:     [B,G,2D]  (concat(mean, max))
    """
    assert pooling in ("mean", "max", "meanmax"), f"Unknown pooling={pooling}"

    B, T, D = F_btd.shape
    _, G, _ = ranges_bG2.shape

    pooled_rows = []
    for b in range(B):
        Fb = F_btd[b]       # [T,D]
        rb = ranges_bG2[b]  # [G,2]
        feats = []
        for i in range(G):
            t0 = int(rb[i, 0].item())
            t1 = int(rb[i, 1].item())

            # clamp
            t0 = max(0, min(T - 1, t0))
            t1 = max(0, min(T - 1, t1))
            if t1 < t0:
                t0, t1 = t1, t0

            seg = Fb[t0: t1 + 1, :]  # [L,D]
            mean_v = paddle.mean(seg, axis=0)  # [D]

            if pooling == "mean":
                v = mean_v
            else:
                max_v = paddle.max(seg, axis=0)  # [D]
                if pooling == "max":
                    v = max_v
                else:
                    v = paddle.concat([mean_v, max_v], axis=0)  # [2D]

            feats.append(v)

        pooled_rows.append(paddle.stack(feats, axis=0))  # [G,D] or [G,2D]

    return paddle.stack(pooled_rows, axis=0)  # [B,G,D] or [B,G,2D]


# -----------------------
# Robust head shape inference
# -----------------------
def _infer_head_structure_from_state(state: Dict[str, paddle.Tensor]) -> Tuple[int, int, int, bool]:
    """
    PaddlePaddle Linear weight shape is [in_features, out_features].

    Infer (in_dim, hidden, num_fonts, is_mlp) from head state dict.
    Supports:
      - Linear only: net.weight [in_dim, num_fonts]
      - MLP Sequential: net.0.weight [in_dim, hidden], net.<last>.weight [hidden, num_fonts]
    """
    # Linear-only head
    if "net.weight" in state:
        w = state["net.weight"]  # [in_dim, num_fonts]
        in_dim = int(w.shape[0])
        num_fonts = int(w.shape[1])
        hidden = 0
        return in_dim, hidden, num_fonts, False

    # MLP head: gather net.<i>.weight keys
    weight_keys = [k for k in state.keys() if k.startswith("net.")
                   and k.endswith(".weight")]
    if not weight_keys:
        raise RuntimeError(
            f"No net.*.weight keys found. Keys: {list(state.keys())[:15]} ...")

    def layer_idx(k: str) -> int:
        return int(k.split(".")[1])  # "net.3.weight" -> 3

    weight_keys_sorted = sorted(weight_keys, key=layer_idx)
    first_k = weight_keys_sorted[0]
    last_k = weight_keys_sorted[-1]

    first_w = state[first_k]  # [in_dim, hidden]
    last_w = state[last_k]    # [hidden, num_fonts]

    in_dim = int(first_w.shape[0])
    hidden = int(first_w.shape[1])
    num_fonts = int(last_w.shape[1])

    # Sanity: last_w in_features should match hidden
    if int(last_w.shape[0]) != hidden:
        hidden = int(last_w.shape[0])  # trust last layer

    return in_dim, hidden, num_fonts, True

# -----------------------
# Head definitions (match common training)
# -----------------------


class FontHeadLinear(nn.Layer):
    def __init__(self, in_dim: int, num_fonts: int):
        super().__init__()
        self.net = nn.Linear(in_dim, num_fonts)

    def forward(self, x_bgd: paddle.Tensor) -> paddle.Tensor:
        return self.net(x_bgd)


class FontHeadMLP(nn.Layer):
    """
    Matches your training-style Sequential:
      Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(self, in_dim: int, hidden: int, num_fonts: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_fonts),
        )

    def forward(self, x_bgd: paddle.Tensor) -> paddle.Tensor:
        return self.net(x_bgd)


# -----------------------
# Confusion + metrics
# -----------------------
def _confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, Any]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    support = cm.sum(axis=1)
    correct = np.diag(cm)
    total = cm.sum()

    acc_micro = float(correct.sum() / max(1, total))

    acc_per_class = np.divide(
        correct, np.maximum(1, support), dtype=np.float64)
    acc_macro = float(acc_per_class.mean()) if num_classes > 0 else 0.0

    pred_count = cm.sum(axis=0)
    precision = np.divide(correct, np.maximum(1, pred_count), dtype=np.float64)
    recall = np.divide(correct, np.maximum(1, support), dtype=np.float64)
    f1 = np.divide(2 * precision * recall, np.maximum(1e-12,
                   precision + recall), dtype=np.float64)

    return {
        "cm": cm,
        "acc_micro": acc_micro,
        "acc_macro": acc_macro,
        "acc_per_class": acc_per_class,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "total": int(total),
        "correct": int(correct.sum()),
    }


@paddle.no_grad()
def run_report(
    rec_model: nn.Layer,
    head: nn.Layer,
    dl: paddle.io.DataLoader,
    pooling: str,
    num_fonts: int,
) -> Dict[str, Any]:
    head.eval()
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for x, ranges, y, mask in dl:
        F_btd, _ = rec_model.forward_features(x)  # [B,T,D]
        pooled = pool_by_ranges(F_btd, ranges, pooling=pooling)  # [B,G,in_dim]
        logits = head(pooled)  # [B,G,C]
        pred = logits.argmax(axis=-1)  # [B,G]

        m = (mask > 0.5).astype("int64")  # [B,G]

        y_flat = y.numpy().reshape(-1)
        p_flat = pred.numpy().reshape(-1)
        m_flat = m.numpy().reshape(-1).astype(bool)

        y_true_all.extend(y_flat[m_flat].tolist())
        y_pred_all.extend(p_flat[m_flat].tolist())

    y_true = np.asarray(y_true_all, dtype=np.int64)
    y_pred = np.asarray(y_pred_all, dtype=np.int64)

    metrics = _confusion_and_metrics(y_true, y_pred, num_classes=num_fonts)
    metrics["n_tokens"] = int(len(y_true))
    return metrics


def save_confusion_csv(cm: np.ndarray, id2font: Dict[int, str], out_csv: Path) -> None:
    import csv
    labels = [id2font[i] for i in range(cm.shape[0])]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i, lab in enumerate(labels):
            w.writerow([lab] + cm[i].tolist())


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-align-jsonl", type=Path, required=True)
    ap.add_argument("--font-vocab", type=Path, required=True)

    ap.add_argument("--rec-config", type=str, required=True)
    ap.add_argument("--rec-checkpoint", type=str, required=True)
    ap.add_argument("--font-head", type=str, required=True)

    ap.add_argument("--pooling", type=str, default="meanmax",
                    choices=["mean", "max", "meanmax"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="gpu",
                    choices=["gpu", "cpu"])
    ap.add_argument("--rec-image-shape", type=str, default="3,32,320")
    ap.add_argument("--max-graphemes", type=int, default=120)

    ap.add_argument("--out-dir", type=Path, default=None,
                    help="If set, saves confusion_matrix.csv + report.json")
    ap.add_argument("--topk", type=int, default=10,
                    help="Top confusions to print")
    ap.add_argument("--head-dropout", type=float, default=0.1,
                    help="Dropout used in MLP head (for structure only)")
    args = ap.parse_args()

    paddle.set_device("cpu" if args.device == "cpu" else "gpu")

    vocab = json.loads(args.font_vocab.read_text(encoding="utf-8"))
    font2id = vocab["font2id"]
    id2font = {int(v): k for k, v in font2id.items()}
    num_fonts_vocab = int(vocab["num_fonts"])

    rec_shape = tuple(int(x.strip()) for x in args.rec_image_shape.split(","))
    ds = AlignJsonlFontDataset(
        align_jsonl=args.val_align_jsonl,
        font2id=font2id,
        rec_image_shape=rec_shape,
        max_graphemes=args.max_graphemes,
    )
    dl = paddle.io.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_font_batch,
        drop_last=False,
        num_workers=2,
    )

    # Load frozen OCR model
    rec_model = load_rec_model_with_features(
        config_path=args.rec_config,
        checkpoint_path=args.rec_checkpoint,
        device=args.device,
    )
    rec_model.eval()
    for p in rec_model.parameters():
        p.stop_gradient = True

    # Load head checkpoint & infer exact shape
    head_state = paddle.load(args.font_head)
    in_dim, hidden, num_fonts_head, is_mlp = _infer_head_structure_from_state(
        head_state)

    # Enforce vocab consistency (avoid silent nonsense)
    if num_fonts_head != num_fonts_vocab:
        raise RuntimeError(
            f"Vocab num_fonts={num_fonts_vocab} but head checkpoint num_fonts={num_fonts_head}. "
            "Use the SAME font_vocab.json that was used during training."
        )

    # Build the exact head structure and load weights
    if is_mlp:
        head = FontHeadMLP(in_dim=in_dim, hidden=hidden,
                           num_fonts=num_fonts_head, dropout=args.head_dropout)
    else:
        head = FontHeadLinear(in_dim=in_dim, num_fonts=num_fonts_head)

    # strict load: will crash if mismatch (good)
    head.set_state_dict(head_state)
    head.eval()

    # Run report
    metrics = run_report(rec_model, head, dl,
                         pooling=args.pooling, num_fonts=num_fonts_head)
    cm = metrics.pop("cm")

    print("\n=== VALIDATION REPORT ===")
    print(f"pooling={args.pooling}  head={'MLP' if is_mlp else 'Linear'}  in_dim={in_dim} hidden={hidden} num_fonts={num_fonts_head}")
    print(
        f"tokens={metrics['n_tokens']}  correct={metrics['correct']}  total={metrics['total']}")
    print(
        f"acc_micro={metrics['acc_micro']:.4f}  acc_macro={metrics['acc_macro']:.4f}")

    acc_pc = metrics["acc_per_class"]
    prec = metrics["precision"]
    rec = metrics["recall"]
    f1 = metrics["f1"]
    sup = metrics["support"]

    print("\n--- Per-class ---")
    for i in range(num_fonts_head):
        name = id2font.get(i, f"class_{i}")
        print(
            f"{i:2d} {name:20s} "
            f"support={int(sup[i]):6d}  "
            f"acc={acc_pc[i]:.4f}  "
            f"P={prec[i]:.4f}  R={rec[i]:.4f}  F1={f1[i]:.4f}"
        )

    print(f"\n--- Top {args.topk} confusions (true -> pred) ---")
    conf_pairs = []
    for t in range(num_fonts_head):
        for p in range(num_fonts_head):
            if t == p:
                continue
            c = int(cm[t, p])
            if c > 0:
                conf_pairs.append((c, t, p))
    conf_pairs.sort(reverse=True)
    for c, t, p in conf_pairs[: args.topk]:
        tn = id2font.get(t, str(t))
        pn = id2font.get(p, str(p))
        print(f"{tn} -> {pn}: {c}")

    # Save outputs
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = args.out_dir / "confusion_matrix.csv"
        save_confusion_csv(cm, id2font, out_csv)

        out_json = args.out_dir / "report.json"
        to_save = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in metrics.items()
        }
        out_json.write_text(json.dumps(to_save, indent=2), encoding="utf-8")

        print(f"\n[OK] wrote {out_csv}")
        print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
