"""
train.py — AIEEEs MTC-AIC4 Phase I
====================================
NOTE: This tracker does NOT require training.

The feature extractor uses MobileNet-V3-Small with standard ImageNet
pretrained weights loaded automatically by torchvision at runtime.
No fine-tuning was performed and no learned checkpoints exist.

The submission is purely classical:
  - CSRT handles frame-to-frame localisation
  - MobileNet-V3-Small (frozen ImageNet weights) provides appearance verification

To run inference, use track.py directly — no training step is needed.
See README.md for the exact inference command.
"""


def main():
    print(
        "This tracker requires no training.\n"
        "MobileNet-V3-Small ImageNet weights are loaded automatically by torchvision.\n"
        "Run inference directly with:\n\n"
        "    python track.py \\\n"
        "        --dataset_root /path/to/contest_release \\\n"
        "        --save_root    results \\\n"
        "        --config       configs/tracker_config.yaml \\\n"
        "        --generate_csv\n"
    )


if __name__ == "__main__":
    main()
