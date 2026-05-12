"""
Inference — predict tumor class + visualize attention maps
==========================================================
Usage:
  python predict.py --image mri.jpg  --checkpoint outputs/best_model.pth
  python predict.py --image folder/  --checkpoint outputs/best_model.pth --show_attention
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import build_model
from utils  import load_checkpoint, visualize_attention


CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_LABELS = {
    "glioma":     "Glioma",
    "meningioma": "Meningioma",
    "no_tumor":   "No Tumor",
    "pituitary":  "Pituitary Tumor",
}
CLASS_COLORS = {
    "glioma":     "#e74c3c",
    "meningioma": "#e67e22",
    "no_tumor":   "#27ae60",
    "pituitary":  "#8e44ad",
}


def predict_single(model, image_path, transform, device, class_names):
    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx   = probs.argmax()
    pred_class = class_names[pred_idx]
    return {
        "image": image, "pred_class": pred_class,
        "confidence": float(probs[pred_idx] * 100),
        "probs": {c: float(p * 100) for c, p in zip(class_names, probs)},
    }


def visualize_prediction(result, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Brain Tumor ViT Classification", fontsize=14, fontweight="bold")

    axes[0].imshow(result["image"])
    axes[0].axis("off")
    pred  = result["pred_class"]
    color = CLASS_COLORS.get(pred, "#333")
    axes[0].set_title(
        f"Prediction: {CLASS_LABELS.get(pred, pred)}\nConfidence: {result['confidence']:.1f}%",
        color=color, fontweight="bold", fontsize=12
    )

    classes     = list(result["probs"].keys())
    probs       = list(result["probs"].values())
    bar_colors  = [CLASS_COLORS.get(c, "#95a5a6") for c in classes]
    bars = axes[1].barh(classes, probs, color=bar_colors, edgecolor="white")
    axes[1].set_xlim(0, 110)
    axes[1].set_xlabel("Probability (%)")
    axes[1].set_title("Class Probabilities")
    axes[1].grid(axis="x", alpha=0.3)
    for bar, p in zip(bars, probs):
        axes[1].text(bar.get_width() + 0.5,
                     bar.get_y() + bar.get_height() / 2,
                     f"{p:.1f}%", va="center")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = build_model(args.arch, num_classes=len(CLASS_NAMES), pretrained=False)
    ckpt  = load_checkpoint(model, args.checkpoint, device)
    class_names = ckpt.get("class_names") or CLASS_NAMES

    image_path = Path(args.image)
    images = [image_path] if image_path.is_file() else sorted(
        p for p in image_path.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Found {len(images)} image(s).\n")

    os.makedirs(args.save_dir, exist_ok=True)

    for img_path in images:
        result = predict_single(model, img_path, transform, device, class_names)
        print(f"{img_path.name}:")
        print(f"  → {CLASS_LABELS.get(result['pred_class'], result['pred_class'])}"
              f"  ({result['confidence']:.1f}%)")
        for cls, p in result["probs"].items():
            print(f"     {cls:15s}: {p:5.1f}%")
        print()

        # Standard prediction visualization
        pred_out = os.path.join(args.save_dir, f"{img_path.stem}_pred.png")
        visualize_prediction(result, save_path=pred_out)

        # Attention rollout (ViT / Hybrid only — skips gracefully on Swin)
        if args.show_attention:
            attn_out = os.path.join(args.save_dir, f"{img_path.stem}_attention.png")
            try:
                visualize_attention(
                    model, str(img_path), class_names,
                    img_size=args.img_size, save_path=attn_out, device=device
                )
            except Exception as e:
                print(f"  Attention viz skipped: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image",          required=True)
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--arch",           default="swin_base",
                   choices=["vit_b16","swin_base","hybrid_vit"])
    p.add_argument("--img_size",       type=int, default=224)
    p.add_argument("--save_dir",       default="outputs/predictions")
    p.add_argument("--show_attention", action="store_true",
                   help="Generate attention rollout overlays")
    args = p.parse_args()
    main(args)
