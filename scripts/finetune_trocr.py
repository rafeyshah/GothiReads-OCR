#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# ----------------------------
# Manifest loading (same logic style as your harness/eval)
# ----------------------------


def load_manifest(csv_path: str, limit: Optional[int] = None) -> Tuple[List[str], List[str], List[str]]:
    ids, imgs, gts = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("ok") != "TRUE":
                continue
            img, txt = r.get("image_path"), r.get("txt_path")
            if not img or not txt:
                continue
            txt_p = Path(txt)
            if not txt_p.is_file():
                continue
            gt = txt_p.read_text(encoding="utf-8").strip()
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
def build_augmentor(use_albu: bool = True):
    if not use_albu:
        return None
    try:
        import albumentations as A
        import cv2
    except Exception:
        return None

    # light but helpful: blur, brightness/contrast, tiny rotate/affine, very light elastic
    aug = A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
        ], p=0.25),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.35),
        A.Affine(rotate=(-1, 1), shear=(-1, 1),
                 translate_percent=(0, 0.01), p=0.35),
        A.ElasticTransform(alpha=5, sigma=20, alpha_affine=3, p=0.10),
    ])
    return (aug, cv2)


class GothiReadTrOCRDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths: List[str], texts: List[str], processor: TrOCRProcessor, augmentor=None):
        self.img_paths = img_paths
        self.texts = texts
        self.processor = processor
        self.augmentor = augmentor  # (albumentations_compose, cv2)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")

        if self.augmentor is not None:
            aug, cv2 = self.augmentor
            import numpy as np
            arr = np.array(img)
            # albumentations expects BGR sometimes; but for these ops RGB is ok
            out = aug(image=arr)["image"]
            img = Image.fromarray(out)

        # Pixel values
        pixel = self.processor(images=img, return_tensors="pt").pixel_values[0]

        # Labels
        labels = self.processor.tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        # Replace pad token id's by -100 so they are ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--model_name", default="microsoft/trocr-handwritten")
    ap.add_argument("--out_dir", default="runs/trocr_handwritten_ft")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--train_bs", type=int, default=4)
    ap.add_argument("--eval_bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_label_len", type=int, default=128)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--resume_from", type=str, default=None,
                    help="Path to a Trainer checkpoint dir (e.g., runs/.../checkpoint-5000)")
    ap.add_argument("--no_aug", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load data
    _, train_imgs, train_txts = load_manifest(
        args.train_manifest, args.limit_train)
    _, val_imgs, val_txts = load_manifest(args.val_manifest, args.limit_val)
    print(f"Train lines: {len(train_imgs)} | Val lines: {len(val_imgs)}")

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(
        args.model_name,
        use_safetensors=True,
    )

    # Important defaults for TrOCR fine-tuning
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # Decode params you’ll “lock” on Day 10 (start with something reasonable)
    model.generation_config.max_length = args.max_label_len
    model.generation_config.num_beams = 4  # beam-search during eval
    model.generation_config.early_stopping = True

    augmentor = None if args.no_aug else build_augmentor(use_albu=True)

    train_ds = GothiReadTrOCRDataset(
        train_imgs, train_txts, processor, augmentor=augmentor)
    val_ds = GothiReadTrOCRDataset(
        val_imgs, val_txts, processor, augmentor=None)

    # Metrics: CER using jiwer (fallback: naive CER)
    try:
        from jiwer import cer as jiwer_cer

        def compute_metrics(eval_pred):
            preds = eval_pred.predictions
            label_ids = eval_pred.label_ids

            pred_ids = preds.argmax(-1) if preds.ndim == 3 else preds
            pred_str = processor.batch_decode(
                pred_ids, skip_special_tokens=True)

            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            label_str = processor.batch_decode(
                label_ids, skip_special_tokens=True)

            return {"cer": float(jiwer_cer(label_str, pred_str))}
    except Exception:
        def _edit_distance(a: str, b: str) -> int:
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
                    curr[i] = min(prev[i] + 1, curr[i - 1] +
                                  1, prev[i - 1] + cost)
                prev, curr = curr, prev
            return prev[n]

        def compute_metrics(eval_pred):
            preds = eval_pred.predictions
            label_ids = eval_pred.label_ids
            pred_ids = preds.argmax(-1) if preds.ndim == 3 else preds
            pred_str = processor.batch_decode(
                pred_ids, skip_special_tokens=True)
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            label_str = processor.batch_decode(
                label_ids, skip_special_tokens=True)
            total_edits, total_chars = 0, 0
            for gt, pr in zip(label_str, pred_str):
                total_edits += _edit_distance(gt, pr)
                total_chars += max(1, len(gt))
            return {"cer": total_edits / max(1, total_chars)}

    # Training args
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        learning_rate=args.lr,

        # ✅ saving + evaluation
        save_strategy="steps",
        save_steps=500,                 # adjust (e.g., 200/500/1000)
        save_total_limit=3,             # keep last 3 checkpoints

        eval_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=50,

        load_best_model_at_end=True,
        metric_for_best_model="cer",    # or "eval_cer" depending on your compute_metrics key
        greater_is_better=False,

        fp16=True,
        report_to="none",

        eval_accumulation_steps=8,
        predict_with_generate=True,
        generation_max_length=64,
        generation_num_beams=4,
        disable_tqdm=True,

    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)

    # Save best
    print("Saving best model + processor to:", args.out_dir)
    trainer.save_model(args.out_dir)
    processor.save_pretrained(args.out_dir)

    # Save decode config you used
    gen_conf_path = Path(args.out_dir) / "decode_config.txt"
    gen_conf_path.write_text(
        f"max_length={model.generation_config.max_length}\n"
        f"num_beams={model.generation_config.num_beams}\n"
        f"early_stopping={model.generation_config.early_stopping}\n",
        encoding="utf-8",
    )
    print("Wrote", gen_conf_path)


if __name__ == "__main__":
    main()
