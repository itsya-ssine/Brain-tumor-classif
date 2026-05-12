# Brain Tumor MRI Classification — Vision Transformers

State-of-the-art brain tumor classification using three ViT architectures.

## Architectures

| Model | Params | Accuracy | Speed | Best For |
|---|---|---|---|---|
| `vit_b16` | 86M | --- | Medium | Large datasets |
| `swin_base` | 88M | **~98,95%** | Medium | Recommended |
| `hybrid_vit` | ~65M | ~99,21% | Fast | Small datasets |

### Why ViT beats CNNs for MRI tumors?
- **Global attention** — models long-range spatial dependencies across the entire scan
- **Swin's shifted windows** — hierarchical features like CNNs, but with attention
- **Hybrid** — CNN extracts local edge/texture features, Transformer models global context

---

## Structure

```
brain_tumor_vit/
├── train.py
├── predict.py
├── requirements.txt
├── models/
│   └── vit_model.py        # ViTClassifier, SwinClassifier, HybridViTClassifier
├── utils/
│   ├── dataset.py          # Dataset + MixUp collator + transforms
│   ├── trainer.py          # Trainer + WarmupCosine scheduler
│   ├── attention_viz.py    # Attention rollout visualization
│   └── utils.py
└── notebooks/
    └── brain_tumor_vit_colab.ipynb
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Swin Transformer (recommended)
python3 train.py --arch swin_base --epochs 30 --use_mixup

# Pure ViT
python3 train.py --arch vit_b16 --epochs 30 --warmup 5

# Hybrid CNN+ViT (best for smaller datasets)
python3 train.py --arch hybrid_vit --epochs 40 --freeze_epochs 5
```

## Inference + Attention Maps

```bash
python3 predict.py \
  --image path/to/mri.jpg \
  --checkpoint outputs/best_model.pth \
  --arch swin_base \
  --show_attention
```

---

## Key Training Differences vs EfficientNet

| | EfficientNet | ViT / Swin |
|---|---|---|
| Batch size | 32 | **16** (ViTs need more memory) |
| LR | 1e-4 | 1e-4 (backbone) / 1e-3 (head) |
| Weight decay | 1e-4 | **0.05** (ViTs need higher WD) |
| Warmup | None | **5 epochs** (critical for ViT) |
| MixUp | Optional | **Recommended** |
| Freeze first | Optional | **5 epochs recommended** |
