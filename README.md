# 🏛️ Gothi-Read

**Track B:** *OCR + Font Group Recognition (Per-Character Multi-Task)*  
**Author:** Abdul Rafey  
**Repository:** https://github.com/rafeyshah/gothi-read

---

## 🚀 Overview

**Gothi-Read** is an end-to-end OCR + font-group recognition framework developed for **Pattern Recognition Lab**.
The goal is to build and benchmark models capable of:

1. **Optical Character Recognition** — text transcription from scanned lines.
2. **Font Group Recognition** — predicting the font category for every character.

The repository now provides:

* A verified, Unicode-safe data pipeline
* Manifest generation and integrity checks
* Visualization of font annotations
* Evaluation scripts with unified model harness
* Metrics computation for CER/WER and font accuracy

## ⚙️ Environment Setup

* Configured **Python 3 + PyTorch + Hugging Face + CUDA**.
* Verified GPU availability and reproducibility across Colab and VS Code.
* Clear modular directory layout: `scripts/`, `src/`, `notebooks/,` `runs/`.

Main dependencies:

```bash
pip install torch torchvision torchaudio transformers jiwer pillow regex matplotlib
```

## 🧾 Dataset Handling and Validation

* `build_manifest.py` – scans dataset folders to create manifest CSVs listing `.jpg`, `.txt`, and `.font` triplets.
* `check_integrity.py` – summarizes file presence & alignment health.
* `make_test_split.py` – builds reproducible test subsets.
* Verified 100 % length alignment between text and font sequences.

**Validation Integrity Summary**

* Total lines : 4040
* Clean (ok=True) : 3827 (94.73 %)
* Missing txt : 213
* Length mismatches : 0
* ✅ 94.7 % of validation lines are clean — ready for evaluation.

## 🧠 Unified Model Evaluation Harness

`harness.py` provides a single interface to evaluate any OCR model.

**Outputs saved to**

```
runs/<model>/<date>/
  preds.txt  
  metrics.json  
  per_line.csv
```

---

## 📊 OCR Benchmarks

### 🔁 Zero-Shot OCR Baselines

All zero-shot models were evaluated on the same **valid_clean.csv** split using the unified evaluation harness.

| run                         | CER          | WER          |
| --------------------------- | ------------ | ------------ |
| **paddle-ocr-server-rec**   | **0.203298** | **0.755115** |
| **trocr-handwritten-beam**  | 0.356683     | 2.168666     |
| paddle-ocr-mobile-rec       | 1.835468     | 1.411341     |
| trocr-handwritten-greedy    | 0.383486     | 2.331641     |
| trocr-large-printed         | 0.840328     | 5.109445     |
| donut-base-ocr              | 0.596529     | 1.071143     |
| parseq                      | 0.7040       | 0.9927       |
| abinet                      | 0.7576       | 0.9932       |
| vitstr                      | 0.7573       | 0.9934       |

---

### 🧪 Fine-Tuned PaddleOCR

The strongest zero-shot model (**PaddleOCR – server recognizer**) was subsequently **fine-tuned for 5 epochs** on the training split and evaluated on the same `valid_clean.csv` subset.

**Evaluation metrics (lower is better):**

* **CER (Character Error Rate): 0.0128 (~1.3 %)**  
* **WER (Word Error Rate): 0.0796 (~8 %)**

This represents a **substantial improvement over the zero-shot baseline**, achieving high-quality character-level transcription across historical Gothic and mixed-font data. Performance is strongest on single-font lines, with higher (expected) error rates on mixed-font samples.

The best checkpoint was selected by **minimum validation CER**, and decoding parameters were fixed. This fine-tuned PaddleOCR model is retained as a **secondary CTC-based OCR** for later ensembling and character-level font alignment.

---

## 🔜 Next Steps

* Fine-tune **TrOCR-handwritten** and select the primary OCR by lowest CER.
* Add a font-classification head for per-character font prediction.
* Ensemble CTC (PaddleOCR) and seq2seq (TrOCR) models.
* Compute joint **text CER + font-CER** for final evaluation.

## 🏁 Summary

**Gothi-Read** now includes a validated data pipeline, visualization system, and unified model evaluation framework. OCR benchmarking and fine-tuning are complete for PaddleOCR, and the project is ready for multi-model comparison and multi-task font recognition experiments.

