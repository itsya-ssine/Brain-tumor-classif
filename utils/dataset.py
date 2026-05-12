"""
Brain Tumor MRI Dataset — ViT edition
======================================
Identical folder layout to the EfficientNet project:

  data/
    train/  glioma/  meningioma/  no_tumor/  pituitary/
    val/    ...
    test/   ...

Key difference from CNN datasets:
  • ViT is trained at fixed 224×224 (or 384×384 for higher-res fine-tuning)
  • Normalisation uses the same ImageNet stats (mean/std)
  • We add MixUp / CutMix augmentation support for ViT's data-hungry nature
"""

import os
import random
from pathlib import Path
from collections import Counter
from typing import Optional, Callable, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── Recommended transforms ─────────────────────────────────────────────────

def get_vit_transforms(img_size: int = 224, augment_level: str = "medium"):
    """
    Return (train_transform, val_transform) tuned for Vision Transformers.

    augment_level:
        'light'  — safe for small datasets
        'medium' — default, good for ~5k images
        'strong' — for larger datasets / fine-tuning
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if augment_level == "light":
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif augment_level == "strong":
        train_transform = transforms.Compose([
            transforms.Resize((int(img_size * 1.15), int(img_size * 1.15))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])

    else:  # medium
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ])

    return train_transform, val_transform


# ── Dataset ─────────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for Brain Tumor MRI classification.

    Supports optional MixUp collation (set use_mixup=True on train loader).
    """
    CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_names: Optional[list] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = class_names or self._discover_classes()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples, self.targets = self._load_samples()
        print(f"[Dataset] {root_dir}  →  {len(self.samples)} images  |  "
              f"classes: {self.get_class_counts()}")

    # ── internals ──────────────────────────────────────────────────────────

    def _discover_classes(self):
        return sorted(
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def _load_samples(self):
        valid = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        samples, targets = [], []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                print(f"  Warning: folder not found → {cls_dir}")
                continue
            label = self.class_to_idx[cls]
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in valid:
                    samples.append((str(p), label))
                    targets.append(label)
        if not samples:
            raise RuntimeError(f"No images found under {self.root_dir}")
        return samples, targets

    # ── public ─────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> dict:
        c = Counter(self.targets)
        return {self.classes[k]: v for k, v in sorted(c.items())}

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss."""
        counts = Counter(self.targets)
        total  = len(self.targets)
        n_cls  = len(self.classes)
        w = torch.tensor(
            [total / (n_cls * counts[i]) for i in range(n_cls)],
            dtype=torch.float32
        )
        return w / w.sum() * n_cls


# ── MixUp collate ───────────────────────────────────────────────────────────

class MixUpCollator:
    """
    MixUp data augmentation at the batch level.
    Particularly effective for ViT training.

    Usage:
        train_loader = DataLoader(dataset, collate_fn=MixUpCollator(alpha=0.4))
    """
    def __init__(self, alpha: float = 0.4, num_classes: int = 4):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)                         # (B, C, H, W)
        labels = torch.tensor(labels, dtype=torch.long)

        # One-hot
        targets = torch.zeros(len(labels), self.num_classes)
        targets.scatter_(1, labels.unsqueeze(1), 1)

        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(images.size(0))

        mixed_images  = lam * images  + (1 - lam) * images[idx]
        mixed_targets = lam * targets + (1 - lam) * targets[idx]

        return mixed_images, mixed_targets
