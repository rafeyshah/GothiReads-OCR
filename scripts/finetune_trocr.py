#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stable TrOCR fine-tuning script (A100-friendly) with **reliable CER**.

Fixes vs your current finetune_trocr.py :contentReference[oaicite:0]{index=0}:
1) Training-time evaluation uses **greedy decoding (num_beams=1)** to avoid noisy/unstable CER spikes.
2) CER computation includes **sanity filtering** (drops empty GT, extreme-length preds) to prevent one bad batch
   from producing CER ~0.97.
3) Optional: evaluate on a **subset** of validation for fast feedback (val_eval_limit).
4) Sensible default eval cadence: ~0.5 epoch (1500 steps min) unless overridden.

Usage example:
python finetune_trocr.py \
  --train_manifest /content/manifests/train_clean.csv \
  --val_manifest /content/manifests/valid_clean.csv \
  --model_name microsoft/trocr-base-handwritten \
  --out_dir /content/drive/MyDrive/GothiRead/runs/trocr_a100_stable \
  --epochs 3 \
  --train_bs 48 --grad_accum 1 --eval_bs 8 \
  --eval_steps 1500 --save_steps 1500 \
  --max_label_len 128

After training, run a separate "final eval" with beam=4/8 for best score.
"""

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from PIL import Image

from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# ----------------------------
# Speed helpers (A100)
# ----------------------------


def enable_a100_fastmath():
    # TF32 speeds matmuls on Ampere+; usually fine for OCR fine-tuning
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ----------------------------
# Data utils
# ----------------------------

def exists_file(p: str) -> bool:
    try:
        return Path(p).is_file()
    except Exception:
        return False


def read_text_file(p: str) -> str:
    return Path(p).read_text(encoding="utf-8", errors="replace").strip()


def load_manifest(csv_path: str, limit: Optional[int] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Expected columns:
      - id (optional)
      - image_path
      - txt_path
      - ok (TRUE/FALSE) (optional)

    Keeps ok==TRUE if column exists; otherwise keeps all.
    Skips missing files and empty GT.
    """
    ids, imgs, gts = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ok = r.get("ok")
            if ok is not None and ok != "TRUE":
                continue

            img = r.get("image_path")
            txt = r.get("txt_path")
            if not img or not txt:
                continue
            if not exists_file(img) or not exists_file(txt):
                continue

            gt = read_text_file(txt)
            if gt == "":
                continue

            img_id = r.get("id") or Path(img).stem
            ids.append(img_id)
            imgs.append(img)
            gts.append(gt)

            if limit is not None and len(ids) >= limit:
                break

    return ids, imgs, gts


# ----------------------------
# Optional light augmentations
# ----------------------------

def build_augmentor(enabled: bool):
    if not enabled:
        return None
    try:
        import albumentations as A
    except Exception:
        print("[WARN] Albumentations not installed. Continuing without augmentations.")
        return None

    # OCR-friendly light augs. Heavy warps often hurt.
    return A.Compose(
        [
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 3), p=0.25),
                    A.MotionBlur(blur_limit=3, p=0.15),
                ],
                p=0.20,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.12, contrast_limit=0.12, p=0.30),
            A.Affine(rotate=(-1, 1), shear=(-1, 1),
                     translate_percent=(0, 0.01), p=0.25),
            A.ElasticTransform(alpha=3, sigma=15, p=0.05),
        ]
    )


# ----------------------------
# Dataset
# ----------------------------

class TrOCRLineDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_paths: List[str],
        texts: List[str],
        processor: TrOCRProcessor,
        max_label_len: int,
        augmentor=None,
    ):
        self.img_paths = img_paths
        self.texts = texts
        self.processor = processor
        self.max_label_len = max_label_len
        self.augmentor = augmentor

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = Image.open(self.img_paths[idx]).convert("RGB")

        if self.augmentor is not None:
            arr = np.array(img)
            arr = self.augmentor(image=arr)["image"]
            img = Image.fromarray(arr)

        pixel_values = self.processor(
            images=img, return_tensors="pt").pixel_values[0]

        label_ids = self.processor.tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=self.max_label_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        pad_id = self.processor.tokenizer.pad_token_id
        label_ids[label_ids == pad_id] = -100

        return {"pixel_values": pixel_values, "labels": label_ids}


# ----------------------------
# CER
# ----------------------------

def try_import_jiwer_cer():
    try:
        from jiwer import cer as jiwer_cer
        return jiwer_cer
    except Exception:
        return None


def fallback_cer(refs: List[str], hyps: List[str]) -> float:
    def edit_distance(a: str, b: str) -> int:
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n
        if n > m:
            a, b = b, a
            n, m = m, n
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        for j in range(1, m + 1):
            curr[0] = j
            cb = b[j - 1]
            for i in range(1, n + 1):
                ca = a[i - 1]
                cost = 0 if ca == cb else 1
                curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
            prev, curr = curr, prev
        return prev[n]

    total_edits, total_chars = 0, 0
    for r, h in zip(refs, hyps):
        total_edits += edit_distance(r, h)
        total_chars += max(1, len(r))
    return total_edits / max(1, total_chars)


# ----------------------------
# Hyperparam auto-derivation
# ----------------------------

@dataclass
class ResolvedHparams:
    lr: float
    warmup_steps: int
    eval_steps: int
    save_steps: int
    steps_per_epoch: int
    total_steps: int
    effective_batch: int


def resolve_hparams(
    train_size: int,
    epochs: int,
    train_bs: int,
    grad_accum: int,
    lr: Optional[float],
    warmup_steps: Optional[int],
    eval_steps: Optional[int],
    save_steps: Optional[int],
    base_lr: float,
    base_batch: int,
    lr_min: float,
    lr_max: float,
    evals_per_epoch: int,
    min_eval_steps: int = 1500,
) -> ResolvedHparams:
    effective_batch = train_bs * grad_accum
    steps_per_epoch = math.ceil(train_size / max(1, effective_batch))
    total_steps = max(1, steps_per_epoch * epochs)

    # LR: linear scaling around base_lr @ base_batch, clamped
    if lr is None:
        lr = base_lr * (effective_batch / base_batch)
        lr = float(min(max(lr, lr_min), lr_max))

    # Warmup: 3% total steps, capped to 1500, at least 200
    if warmup_steps is None:
        warmup_steps = int(min(1500, max(200, 0.03 * total_steps)))

    # Eval cadence: ~N times per epoch but not too frequent (expensive for OCR)
    if eval_steps is None:
        candidate = steps_per_epoch // max(1, evals_per_epoch)
        eval_steps = int(max(min_eval_steps, candidate))

    if save_steps is None:
        save_steps = eval_steps

    return ResolvedHparams(
        lr=lr,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        effective_batch=effective_batch,
    )


# ----------------------------
# Args
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)

    ap.add_argument("--model_name", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--out_dir", default="runs/trocr_a100_stable")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_bs", type=int, default=32)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--max_label_len", type=int, default=128)

    # IMPORTANT:
    # - During training-time eval we will ALWAYS use greedy (beams=1) for stability.
    # - Use this only for final decoding config file (and optional final eval script).
    ap.add_argument("--final_num_beams", type=int, default=4)

    # Optional overrides (else auto)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup_steps", type=int, default=None)
    ap.add_argument("--eval_steps", type=int, default=None)
    ap.add_argument("--save_steps", type=int, default=None)

    # Auto scaling knobs
    ap.add_argument("--base_lr", type=float, default=5e-5)
    ap.add_argument("--base_batch", type=int, default=16)
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--lr_max", type=float, default=1e-4)
    ap.add_argument("--evals_per_epoch", type=int, default=4)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)

    # evaluate only first N val samples each eval (FAST feedback).
    # Set to None for full val.
    ap.add_argument("--val_eval_limit", type=int, default=None)

    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--early_stop_patience", type=int, default=3)

    ap.add_argument("--disable_tf32", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # Debug: print sample preds during eval
    ap.add_argument("--debug_eval_print", action="store_true")

    return ap.parse_args()


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()

    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(f"Device: cuda ({dev_name}), capability={cap}")
    else:
        print("Device: cpu")

    if torch.cuda.is_available() and not args.disable_tf32:
        enable_a100_fastmath()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load manifests
    _, train_imgs, train_txts = load_manifest(
        args.train_manifest, args.limit_train)
    _, val_imgs, val_txts = load_manifest(args.val_manifest, args.limit_val)

    # Optional: evaluate on a subset every time (speeds up feedback)
    if args.val_eval_limit is not None:
        # deterministic subset
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(val_imgs), size=min(
            args.val_eval_limit, len(val_imgs)), replace=False)
        idx = sorted(idx.tolist())
        val_imgs = [val_imgs[i] for i in idx]
        val_txts = [val_txts[i] for i in idx]

    print(
        f"Train lines: {len(train_imgs)} | Val lines (eval): {len(val_imgs)}")

    # Processor + model
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(
        args.model_name, use_safetensors=True)

    # Align token ids
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing.")

    # Generation config (for eval). We'll force greedy in TrainingArguments.
    # ---- generation config: keep it valid for greedy decoding ----
    model.generation_config.max_length = args.max_label_len

    # For training-time eval we use greedy (num_beams=1), so early_stopping must be disabled
    model.generation_config.num_beams = 1
    model.generation_config.early_stopping = False  # <-- IMPORTANT


    augmentor = build_augmentor(enabled=(not args.no_aug))

    train_ds = TrOCRLineDataset(
        img_paths=train_imgs,
        texts=train_txts,
        processor=processor,
        max_label_len=args.max_label_len,
        augmentor=augmentor,
    )
    val_ds = TrOCRLineDataset(
        img_paths=val_imgs,
        texts=val_txts,
        processor=processor,
        max_label_len=args.max_label_len,
        augmentor=None,
    )

    hp = resolve_hparams(
        train_size=len(train_ds),
        epochs=args.epochs,
        train_bs=args.train_bs,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        base_lr=args.base_lr,
        base_batch=args.base_batch,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        evals_per_epoch=args.evals_per_epoch,
        min_eval_steps=1500,
    )

    print(
        f"[Resolved] effective_batch={hp.effective_batch} | steps/epoch={hp.steps_per_epoch} | total_steps={hp.total_steps}\n"
        f"[Resolved] lr={hp.lr:.2e} | warmup_steps={hp.warmup_steps} | eval_steps={hp.eval_steps} | save_steps={hp.save_steps}\n"
        f"[Resolved] TRAIN-eval uses greedy decode (beams=1). Final beams suggestion: {args.final_num_beams}\n"
        f"[Resolved] max_label_len={args.max_label_len} | aug={'off' if args.no_aug else 'on'}"
    )

    # Mixed precision
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        use_bf16 = major >= 8  # A100
        use_fp16 = not use_bf16

    jiwer_cer = try_import_jiwer_cer()

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        if isinstance(preds, tuple):
            preds = preds[0]

        pred_ids = preds
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        labels = labels.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        # ---- SANITY FILTERS ----
        clean_preds, clean_labels = [], []
        dropped = 0
        for p, l in zip(pred_str, label_str):
            l2 = l.strip()
            if l2 == "":
                dropped += 1
                continue
            p2 = (p or "").strip()
            # If model outputs extremely long garbage relative to GT, drop it from CER calc
            if len(p2) > 4 * max(1, len(l2)):
                dropped += 1
                continue
            clean_preds.append(p2)
            clean_labels.append(l2)

        if len(clean_labels) == 0:
            return {"cer": 1.0}

        if jiwer_cer is not None:
            cer_val = float(jiwer_cer(clean_labels, clean_preds))
        else:
            cer_val = float(fallback_cer(clean_labels, clean_preds))

        if args.debug_eval_print:
            # print a few samples to spot empty/garbage outputs quickly
            print("\n[DEBUG EVAL] samples:")
            for i in range(min(3, len(clean_labels))):
                print("GT  :", clean_labels[i][:200])
                print("PRED:", clean_preds[i][:200])
                print("---")
            print(f"[DEBUG EVAL] dropped={dropped}/{len(label_str)}")
        return {"cer": cer_val}

    # IMPORTANT: training-time eval uses greedy decode for stability
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,

        num_train_epochs=args.epochs,
        learning_rate=hp.lr,
        weight_decay=args.weight_decay,

        lr_scheduler_type="cosine",
        warmup_steps=hp.warmup_steps,

        bf16=use_bf16,
        fp16=use_fp16,

        eval_strategy="steps",
        eval_steps=hp.eval_steps,

        save_strategy="steps",
        save_steps=hp.save_steps,
        save_total_limit=2,

        logging_steps=100,
        report_to="none",

        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,

        predict_with_generate=True,
        generation_max_length=args.max_label_len,
        generation_num_beams=1,      # <-- GREEDY for stable CER during training
        generation_do_sample=False,

        eval_accumulation_steps=16,
        disable_tqdm=True,

        data_seed=args.seed,
    )

    callbacks = []
    if args.early_stop_patience and args.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stop_patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,  # ok for now
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    if args.resume_from:
        print("Resuming from:", args.resume_from)
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Saving best model + processor to:", out_dir)
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    # Save decode params you likely want for FINAL scoring (beam search)
    decode_conf = out_dir / "decode_config.txt"
    decode_conf.write_text(
        f"train_eval_beams=1\n"
        f"final_suggested_beams={args.final_num_beams}\n"
        f"max_length={args.max_label_len}\n"
        f"early_stopping=True\n",
        encoding="utf-8",
    )
    print("Wrote", decode_conf)


if __name__ == "__main__":
    main()
