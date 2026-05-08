"""
train.py — AIEEEs MobileNet Feature Extractor Fine-Tuning
==========================================================
Fine-tunes MobileNet-V3-Small as a Siamese metric learner using
contrastive loss on (template, search) patch pairs extracted from
the aerial tracking datasets.

Usage
-----
    python train.py \
        --dataset_root /data/contest_release \
        --save_dir     checkpoints \
        --epochs       20 \
        --batch_size   32 \
        --lr           1e-4

The best checkpoint is saved to <save_dir>/best_model.pth.
To use it at inference time pass --weights checkpoints/best_model.pth to track.py.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from models.feature_extractor import build_preprocess


# ============================================================
#  Dataset
# ============================================================

class TrackingPatchDataset(Dataset):
    """
    Loads (template, positive_search, negative_search) triplets from
    the aerial tracking datasets.

    A positive pair is two patches of the same object within ±15 frames.
    A negative pair is a patch of the object and a random background crop.
    """

    def __init__(
        self,
        dataset_root: str,
        transform,
        patch_scale: float = 2.0,
        max_pairs_per_seq: int = 200,
    ) -> None:
        super().__init__()
        self.transform          = transform
        self.patch_scale        = patch_scale
        self.triplets: list[tuple[str, str, str]] = []

        self._build_triplets(dataset_root, max_pairs_per_seq)

    # ------------------------------------------------------------------
    def _build_triplets(self, root: str, max_per_seq: int) -> None:
        for dataset_name in sorted(os.listdir(root)):
            dataset_path = os.path.join(root, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            for seq_name in sorted(os.listdir(dataset_path)):
                seq_path = os.path.join(dataset_path, seq_name)
                if not os.path.isdir(seq_path):
                    continue

                # Locate annotation
                anno_file = self._find_anno(dataset_path, seq_name)
                if anno_file is None:
                    continue
                try:
                    anno = np.genfromtxt(anno_file, delimiter=",")
                    if np.isnan(anno).all():
                        anno = np.genfromtxt(anno_file)
                    if anno.ndim == 1:
                        anno = anno.reshape(1, -1)
                except Exception:
                    continue

                images = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))
                if len(images) < 2 or len(anno) < 2:
                    continue

                n = min(len(images), len(anno))
                pairs = min(max_per_seq, n - 1)
                indices = random.sample(range(n - 1), pairs)

                for i in indices:
                    j       = min(i + random.randint(1, 15), n - 1)
                    neg_idx = random.choice([k for k in range(n) if k != i])
                    self.triplets.append((images[i], images[j], images[neg_idx],
                                          tuple(map(int, anno[i][:4])),
                                          tuple(map(int, anno[j][:4])),
                                          tuple(map(int, anno[neg_idx][:4]))))

    # ------------------------------------------------------------------
    @staticmethod
    def _find_anno(dataset_dir: str, seq_name: str) -> str | None:
        base = seq_name
        for suffix in ["_30", "_24", "_20", "_15"]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        for p in [
            os.path.join(dataset_dir, seq_name,    "annotation.txt"),
            os.path.join(dataset_dir, seq_name,    "groundtruth.txt"),
            os.path.join(dataset_dir, "annotation", f"{seq_name}.txt"),
            os.path.join(dataset_dir, "annotation", f"{base}.txt"),
        ]:
            if os.path.exists(p):
                return p
        return None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tmpl_img, pos_img, neg_img, tmpl_box, pos_box, neg_box = self.triplets[idx]

        tmpl_patch = self._crop(cv2.imread(tmpl_img), tmpl_box)
        pos_patch  = self._crop(cv2.imread(pos_img),  pos_box)
        neg_patch  = self._crop_background(cv2.imread(neg_img), neg_box)

        return (
            self.transform(tmpl_patch),
            self.transform(pos_patch),
            self.transform(neg_patch),
        )

    # ------------------------------------------------------------------
    def _crop(self, frame: np.ndarray, box: tuple) -> "PIL.Image.Image":
        from PIL import Image
        if frame is None:
            return Image.new("RGB", (128, 128))
        x, y, w, h = box
        H, W       = frame.shape[:2]
        cx, cy     = x + w // 2, y + h // 2
        half_w     = int(w * self.patch_scale / 2)
        half_h     = int(h * self.patch_scale / 2)
        x1, y1     = max(0, cx - half_w), max(0, cy - half_h)
        x2, y2     = min(W, cx + half_w), min(H, cy + half_h)
        crop       = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return Image.new("RGB", (128, 128))
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def _crop_background(self, frame: np.ndarray, obj_box: tuple) -> "PIL.Image.Image":
        """Return a patch that does NOT overlap the object."""
        from PIL import Image
        if frame is None:
            return Image.new("RGB", (128, 128))
        H, W       = frame.shape[:2]
        x, y, w, h = obj_box
        patch_w    = max(32, w)
        patch_h    = max(32, h)
        for _ in range(20):
            rx = random.randint(0, max(0, W - patch_w))
            ry = random.randint(0, max(0, H - patch_h))
            # Check overlap with the object box
            ix1 = max(rx, x);   iy1 = max(ry, y)
            ix2 = min(rx + patch_w, x + w)
            iy2 = min(ry + patch_h, y + h)
            if ix2 <= ix1 or iy2 <= iy1:           # no overlap
                crop = frame[ry: ry + patch_h, rx: rx + patch_w]
                return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return Image.new("RGB", (128, 128))


# ============================================================
#  Model
# ============================================================

class SiameseFeatureNet(nn.Module):
    """Shared-weight MobileNet-V3-Small with an L2-normalised output."""

    def __init__(self) -> None:
        super().__init__()
        base = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
        )
        base.classifier = nn.Identity()
        self.backbone   = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), p=2, dim=-1)


# ============================================================
#  Loss
# ============================================================

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        pos_dist = 1.0 - (anchor * positive).sum(dim=-1)   # cosine distance
        neg_dist = 1.0 - (anchor * negative).sum(dim=-1)
        loss     = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# ============================================================
#  Training loop
# ============================================================

def train(args: argparse.Namespace) -> None:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_preprocess(tuple(args.resize))

    print(f"Building dataset from {args.dataset_root} ...")
    dataset = TrackingPatchDataset(
        args.dataset_root,
        transform=transform,
        max_pairs_per_seq=args.pairs_per_seq,
    )
    print(f"  {len(dataset):,} training triplets")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model     = SiameseFeatureNet().to(device)
    criterion = TripletLoss(margin=args.margin)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss  = 0.0
        n_batches   = 0

        for tmpl, pos, neg in loader:
            tmpl, pos, neg = tmpl.to(device), pos.to(device), neg.to(device)
            f_tmpl = model(tmpl)
            f_pos  = model(pos)
            f_neg  = model(neg)
            loss   = criterion(f_tmpl, f_pos, f_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if avg_loss < best_loss:
            best_loss   = avg_loss
            ckpt_path   = os.path.join(args.save_dir, "best_model.pth")
            torch.save({"model": model.backbone.state_dict(), "epoch": epoch,
                        "loss": best_loss}, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}")

    # Save final
    final_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save({"model": model.backbone.state_dict(), "epoch": args.epochs,
                "loss": avg_loss}, final_path)
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Final checkpoint: {final_path}")


# ============================================================
#  Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune MobileNet for aerial tracking.")
    p.add_argument("--dataset_root",    required=True)
    p.add_argument("--save_dir",        default="checkpoints")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--margin",          type=float, default=0.3,
                   help="Triplet loss margin.")
    p.add_argument("--pairs_per_seq",   type=int,   default=200,
                   help="Max training pairs per sequence.")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--resize",          type=int,   nargs=2, default=[256, 256],
                   help="Resize (H W) for patches.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
