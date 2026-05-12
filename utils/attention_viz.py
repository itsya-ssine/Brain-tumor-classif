"""
Attention Map Visualisation for Vision Transformers
====================================================
Works with:
  • ViTClassifier  (torchvision vit_b_16)
  • HybridViTClassifier (custom transformer encoder)

Generates "attention rollout" — the product of attention matrices
across all transformer layers, showing which image patches the
model focuses on when making a prediction.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ── Hook-based attention extractor ──────────────────────────────────────────

class AttentionExtractor:
    """
    Registers forward hooks on all MultiheadAttention layers
    and collects attention weights during inference.
    """
    def __init__(self, model):
        self.model    = model
        self.attns    = []
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                h = module.register_forward_hook(self._hook_fn)
                self._handles.append(h)

    def _hook_fn(self, module, inputs, output):
        # output = (attn_output, attn_weights)  — attn_weights shape: (B, N, N)
        if isinstance(output, tuple) and len(output) == 2:
            attn = output[1]
            if attn is not None:
                self.attns.append(attn.detach().cpu())

    def clear(self):
        self.attns = []

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def attention_rollout(attentions: list, discard_ratio: float = 0.9) -> np.ndarray:
    """
    Compute attention rollout from a list of attention matrices.

    Rollout = A_1 @ A_2 @ ... @ A_L  where each A is averaged across heads.
    Returns a 1-D array of length N (number of patches).

    Args:
        attentions   (list[Tensor]): List of (B, H, N, N) or (B, N, N) tensors.
        discard_ratio (float):       Zero-out this fraction of lowest attention values
                                     per layer to reduce noise. Default 0.9.
    Returns:
        np.ndarray: (N,) attention scores for each patch token (excluding CLS).
    """
    result = None

    for attn in attentions:
        # Normalise shape to (N, N)
        if attn.dim() == 4:            # (B, heads, N, N)
            attn = attn[0].mean(dim=0) # → (N, N)
        elif attn.dim() == 3:          # (B, N, N)
            attn = attn[0]             # → (N, N)

        n = attn.shape[-1]

        # Discard low-attention values
        flat = attn.flatten()
        threshold = flat.kthvalue(int(discard_ratio * flat.numel())).values
        attn = torch.where(attn >= threshold, attn, torch.zeros_like(attn))

        # Add residual connection (identity)
        I    = torch.eye(n)
        attn = (attn + I) / 2
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        result = attn if result is None else torch.matmul(attn, result)

    if result is None:
        return np.ones(1)

    # CLS token attends to all patch tokens
    mask = result[0, 1:].numpy()          # shape: (N_patches,)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def visualize_attention(
    model,
    image_path: str,
    class_names: list,
    img_size: int = 224,
    save_path: str = None,
    device=None,
):
    """
    Run the model on a single image and overlay the attention map.

    Args:
        model       : ViTClassifier or HybridViTClassifier
        image_path  : Path to input MRI image
        class_names : List of class name strings
        img_size    : Model input size (default 224)
        save_path   : If given, save figure here; else display inline
        device      : torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    raw_image = Image.open(image_path).convert("RGB")
    tensor    = preprocess(raw_image).unsqueeze(0).to(device)

    # Extract attentions via hooks
    extractor = AttentionExtractor(model)
    with torch.no_grad():
        logits = model(tensor)
    probs      = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx   = probs.argmax()
    pred_class = class_names[pred_idx] if class_names else str(pred_idx)
    confidence = probs[pred_idx] * 100

    # Compute rollout
    grid_size = img_size // 16         # 14 for 224px
    mask = attention_rollout(extractor.attns, discard_ratio=0.85)

    # Reshape to spatial grid
    n_expected = grid_size * grid_size
    if len(mask) >= n_expected:
        mask = mask[:n_expected].reshape(grid_size, grid_size)
    else:
        side = int(math.sqrt(len(mask)))
        mask = mask[:side*side].reshape(side, side)

    # Upsample to original image size
    mask_t = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
    mask_up = F.interpolate(mask_t, size=(img_size, img_size), mode="bilinear",
                            align_corners=False).squeeze().numpy()

    extractor.remove_hooks()

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Attention Rollout — Prediction: {pred_class}  ({confidence:.1f}%)",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(raw_image.resize((img_size, img_size)), cmap="gray")
    axes[0].set_title("Original MRI"); axes[0].axis("off")

    axes[1].imshow(mask_up, cmap="jet")
    axes[1].set_title("Attention Map"); axes[1].axis("off")

    # Overlay
    axes[2].imshow(raw_image.resize((img_size, img_size)), cmap="gray")
    axes[2].imshow(mask_up, alpha=0.5, cmap="jet")
    axes[2].set_title("Overlay"); axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Attention map saved → {save_path}")
    else:
        plt.show()
    plt.close()

    return {"pred_class": pred_class, "confidence": confidence, "probs": probs}
