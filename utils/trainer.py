"""
Trainer for Vision Transformer Brain Tumor Classification
=========================================================
Features:
  • Linear warmup + cosine annealing LR schedule
  • Supports both hard labels and soft MixUp labels
  • Attention rollout visualisation (ViT / Hybrid)
  • Confusion matrix + per-class metrics
  • Best-model checkpointing
"""

import os
import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────
# LR Schedule: linear warmup → cosine annealing
# ─────────────────────────────────────────────────────────────
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linearly increases LR from 0 to base_lr over `warmup_steps`,
    then follows cosine decay to `min_lr`.
    ViTs are sensitive to LR spikes at the start — warmup helps a lot.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (e - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale = self.min_lr + 0.5 * (1 - self.min_lr) * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ─────────────────────────────────────────────────────────────
# Loss: supports both hard labels (CrossEntropy)
#       and soft MixUp labels (KL divergence)
# ─────────────────────────────────────────────────────────────
class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=-1)
        if targets.dim() == 1:                        # hard labels
            return F.nll_loss(log_prob, targets)
        else:                                          # soft labels (MixUp)
            return -(targets * log_prob).sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────
class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        save_dir="outputs",
        class_names=None,
        use_mixup=False,
    ):
        self.model       = model
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device
        self.save_dir    = save_dir
        self.class_names = class_names or []
        self.use_mixup   = use_mixup
        self.criterion   = SoftTargetCrossEntropy()
        os.makedirs(save_dir, exist_ok=True)

        self.history = {k: [] for k in
                        ["train_loss", "val_loss", "train_acc", "val_acc", "lr"]}
        self.best_val_acc = 0.0

    # ── public ─────────────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader, epochs: int = 30):
        print(f"\n{'='*65}")
        print(f"  Training {self.model.__class__.__name__} "
              f"for {epochs} epochs  |  device: {self.device}")
        print(f"{'='*65}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._train_epoch(train_loader)
            vl_loss, vl_acc, preds, labels = self._val_epoch(val_loader)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            for k, v in zip(
                ["train_loss","val_loss","train_acc","val_acc","lr"],
                [tr_loss, vl_loss, tr_acc, vl_acc, lr]
            ):
                self.history[k].append(v)

            marker = ""
            if vl_acc > self.best_val_acc:
                self.best_val_acc = vl_acc
                self._save_checkpoint(epoch, vl_acc)
                marker = "  ✓ saved"

            print(
                f"Ep {epoch:03d}/{epochs}  "
                f"| TrLoss {tr_loss:.4f}  TrAcc {tr_acc:.2f}%"
                f"  | VlLoss {vl_loss:.4f}  VlAcc {vl_acc:.2f}%"
                f"  | LR {lr:.2e}  | {time.time()-t0:.0f}s{marker}"
            )

        print(f"\n  Best Val Accuracy: {self.best_val_acc:.2f}%")
        self._plot_curves()
        self._final_report(val_loader)

    # ── private ────────────────────────────────────────────────────────────

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = correct = total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            if labels.dim() == 2:              # MixUp: use argmax of soft labels
                hard = labels.argmax(dim=1)
            else:
                hard = labels
            correct += (preds == hard).sum().item()
            total   += images.size(0)

        return total_loss / total, 100.0 * correct / total

    def _val_epoch(self, loader):
        self.model.eval()
        total_loss = correct = total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                hard  = labels.argmax(dim=1) if labels.dim() == 2 else labels
                correct += (preds == hard).sum().item()
                total   += images.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(hard.cpu().numpy())

        return total_loss / total, 100.0 * correct / total, all_preds, all_labels

    def _save_checkpoint(self, epoch, val_acc):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "class_names": self.class_names,
            "architecture": self.model.__class__.__name__,
        }, os.path.join(self.save_dir, "best_model.pth"))

    def _plot_curves(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Training History — {self.model.__class__.__name__}",
            fontsize=14, fontweight="bold"
        )

        for ax, (tr_key, vl_key), title, ylabel in zip(
            axes,
            [("train_loss","val_loss"), ("train_acc","val_acc"), ("lr","lr")],
            ["Loss", "Accuracy (%)", "Learning Rate"],
            ["Loss", "Acc (%)", "LR"],
        ):
            if tr_key == "lr":
                ax.plot(epochs, self.history["lr"], "g-", linewidth=2)
                ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                continue
            ax.plot(epochs, self.history[tr_key], "b-o", markersize=3, label="Train")
            ax.plot(epochs, self.history[vl_key], "r-o", markersize=3, label="Val")
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(self.save_dir, "training_curves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Curves saved → {out}")

    def _final_report(self, loader):
        _, _, preds, labels = self._val_epoch(loader)
        print("\n" + "="*65)
        print("CLASSIFICATION REPORT")
        print("="*65)
        print(classification_report(
            labels, preds,
            target_names=self.class_names or None
        ))

        cm = confusion_matrix(labels, preds)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=axes[0])
        axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
        axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")

        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=axes[1])
        axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
        axes[1].set_ylabel("True"); axes[1].set_xlabel("Predicted")

        plt.tight_layout()
        out = os.path.join(self.save_dir, "confusion_matrix.png")
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Confusion matrix saved → {out}\n")
