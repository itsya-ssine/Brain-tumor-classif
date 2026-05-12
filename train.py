"""
Brain Tumor MRI Classification — Vision Transformer Training
=============================================================
Supports three architectures selectable via --arch:

  vit_b16     Pure ViT-B/16 (ImageNet pretrained)
  swin_base   Swin Transformer Base   ← recommended
  hybrid_vit  ResNet-50 + custom Transformer encoder

Usage:
  python train.py --arch swin_base --epochs 30
  python train.py --arch vit_b16   --epochs 30 --img_size 224
  python train.py --arch hybrid_vit --epochs 40 --use_mixup
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from models import build_model, model_summary
from utils  import (
    BrainTumorDataset, get_vit_transforms, MixUpCollator,
    Trainer, WarmupCosineScheduler, set_seed
)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Transforms ─────────────────────────────────────────────────────────
    train_tf, val_tf = get_vit_transforms(args.img_size, augment_level=args.augment)

    # ── Datasets ───────────────────────────────────────────────────────────
    train_ds = BrainTumorDataset(os.path.join(args.data_dir, "train"), train_tf)
    val_ds   = BrainTumorDataset(os.path.join(args.data_dir, "val"),   val_tf)

    # ── DataLoaders ────────────────────────────────────────────────────────
    collate_fn = MixUpCollator(alpha=0.4, num_classes=len(train_ds.classes)) \
                 if args.use_mixup else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(
        architecture=args.arch,
        num_classes=len(train_ds.classes),
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    model_summary(model, name=args.arch)
    model = model.to(device)

    # Optional: freeze backbone for first N epochs (transfer learning phase)
    if args.freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        print(f"Backbone frozen for first {args.freeze_epochs} epochs.\n")

    # ── Optimizer ──────────────────────────────────────────────────────────
    # ViTs benefit from layer-wise learning rate decay
    # Simple version: separate param groups for backbone vs head
    head_params     = [p for n, p in model.named_parameters()
                       if any(k in n for k in ["head","heads","classifier"])]
    backbone_params = [p for n, p in model.named_parameters()
                       if not any(k in n for k in ["head","heads","classifier"])]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr,        "weight_decay": args.wd},
        {"params": head_params,     "lr": args.lr * 10,   "weight_decay": 0.0},
    ])

    # ── Scheduler: warmup + cosine ──────────────────────────────────────────
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup,
        total_epochs=args.epochs,
        min_lr=1e-6
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        class_names=train_ds.classes,
        use_mixup=args.use_mixup,
    )

    # Two-stage training: frozen backbone → then unfreeze
    if args.freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        print(f"── Stage 1: frozen backbone ({args.freeze_epochs} epochs) ──")
        trainer.fit(train_loader, val_loader, epochs=args.freeze_epochs)

        print(f"\n── Stage 2: full fine-tuning ({args.epochs - args.freeze_epochs} epochs) ──")
        model.unfreeze_backbone()
        # Lower LR for fine-tuning
        for g in optimizer.param_groups:
            g["lr"] *= 0.1
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=2,
            total_epochs=args.epochs - args.freeze_epochs,
        )
        trainer.scheduler = scheduler
        trainer.fit(train_loader, val_loader,
                    epochs=args.epochs - args.freeze_epochs)
    else:
        trainer.fit(train_loader, val_loader, epochs=args.epochs)

    print(f"\nDone. Best model → {args.save_dir}/best_model.pth")


if __name__ == "__main__":
    p = argparse.ArgumentParser("ViT Brain Tumor Training")
    p.add_argument("--arch",         default="swin_base",
                   choices=["vit_b16", "swin_base", "hybrid_vit"])
    p.add_argument("--data_dir",     default="data")
    p.add_argument("--save_dir",     default="outputs")
    p.add_argument("--img_size",     type=int,   default=224)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--warmup",       type=int,   default=5,
                   help="Linear warmup epochs (ViTs need this)")
    p.add_argument("--freeze_epochs",type=int,   default=5,
                   help="Epochs with frozen backbone before fine-tuning")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--wd",           type=float, default=0.05,
                   help="Weight decay (AdamW). ViTs prefer higher WD ~0.05")
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--augment",      default="medium",
                   choices=["light","medium","strong"])
    p.add_argument("--use_mixup",    action="store_true",
                   help="Enable MixUp data augmentation")
    p.add_argument("--pretrained",   action="store_true", default=True)
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()
    main(args)
