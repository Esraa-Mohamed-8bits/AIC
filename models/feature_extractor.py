"""
models/feature_extractor.py
----------------------------
Lightweight feature extractor based on MobileNet-V3-Small.
The classifier head is replaced with an Identity layer so that
the network outputs a 576-dimensional L2-normalised embedding.

Model stats (pretrained, no fine-tuning):
  Parameters : ~2.54 M
  FLOPs      : ~0.06 GFLOPs (per 256×256 patch)
  Disk size  : ~9.8 MB (fp32 state-dict)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Pre-processing pipeline (shared across all instances)
# ---------------------------------------------------------------------------
_DEFAULT_RESIZE = (256, 256)
_IMAGENET_MEAN  = [0.485, 0.456, 0.406]
_IMAGENET_STD   = [0.229, 0.224, 0.225]


def build_preprocess(resize_dim: tuple[int, int] = _DEFAULT_RESIZE) -> T.Compose:
    return T.Compose([
        T.Resize(resize_dim),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """
    MobileNet-V3-Small with the classifier replaced by Identity.
    Returns L2-normalised feature vectors.

    Args:
        weights_path: Optional path to a fine-tuned state-dict (.pth).
                      When None the ImageNet pretrained weights are used.
        resize_dim:   (H, W) to resize patches before inference.
        device:       torch.device to run on.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        resize_dim: tuple[int, int] = _DEFAULT_RESIZE,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build backbone
        base = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
        )
        base.classifier = nn.Identity()
        self.backbone = base

        # Load fine-tuned weights when provided
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu")
            # Support both raw state-dicts and checkpoints with a "model" key
            if "model" in state:
                state = state["model"]
            self.backbone.load_state_dict(state, strict=False)

        self.backbone = self.backbone.to(self.device).eval()
        self.preprocess = build_preprocess(resize_dim)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor: (B, 3, H, W) already pre-processed.
        Returns:
            (B, D) L2-normalised embeddings.
        """
        feats = self.backbone(tensor)
        return F.normalize(feats, p=2, dim=-1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract(self, patch_bgr: np.ndarray) -> torch.Tensor | None:
        """
        Convenience wrapper: accepts a raw BGR patch (H, W, 3).
        Returns a 1-D embedding tensor or None on invalid input.
        """
        if patch_bgr is None or patch_bgr.shape[0] < 5 or patch_bgr.shape[1] < 5:
            return None
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        img    = Image.fromarray(patch_rgb)
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        return self.forward(tensor).squeeze(0)

    # ------------------------------------------------------------------
    @staticmethod
    def embedding_dim() -> int:
        """Output dimension of the MobileNet-V3-Small feature vector."""
        return 576


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    extractor = FeatureExtractor()
    dummy_bgr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    # Warm-up
    for _ in range(3):
        extractor.extract(dummy_bgr)

    t0    = time.perf_counter()
    iters = 50
    for _ in range(iters):
        feat = extractor.extract(dummy_bgr)
    elapsed_ms = (time.perf_counter() - t0) / iters * 1000

    print(f"Embedding dim : {feat.shape[0]}")
    print(f"Avg latency   : {elapsed_ms:.2f} ms / patch")
    print(f"Device        : {extractor.device}")
