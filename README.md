# AIEEEs — MTC-AIC4 Phase I Submission

**Team:** AIEEEs  
**Competition:** MTC-AIC4 (Military Technical College Artificial Intelligence Competition)  
**Task:** Single-Object Aerial Tracking

---

## No Checkpoint Required

This tracker is **purely classical** — it uses:
- **CSRT** for frame-to-frame localisation (OpenCV)
- **MobileNet-V3-Small** with **frozen ImageNet pretrained weights** for appearance verification

**No fine-tuning was performed. No learned checkpoint exists.**  
Weights are downloaded automatically by torchvision on first run.

---

## Method Overview

Our tracker combines a **CSRT backbone** with a **MobileNet-V3-Small appearance model** operating in a tight latency budget:

- **Feature Extractor:** MobileNet-V3-Small (ImageNet pretrained, **no fine-tuning**). Outputs a 576-dimensional L2-normalised embedding.
- **Template Bank:** Maintains up to 8 diverse appearance templates, preserving the original for robustness against drift.
- **Velocity Estimator:** 6-frame rolling-window velocity for Kalman-style motion prediction.
- **Scale Controller:** Clamps per-frame scale changes to ≤ 8 %, preventing runaway box growth.
- **Out-of-View Manager:** Remembers exit borders to prioritise border sweeps on re-entry.
- **Object-Size Profiles:** Three profiles (tiny / medium / large) tune grid size, search throttle, and whether MobileNet is used at all — keeping tiny-object frames within the 25 ms budget.
- **Dynamic Reinit Thresholds:** The CSRT reinit confidence threshold drops progressively the longer the target is lost, enabling recovery in challenging sequences.

### Efficiency

| Metric | Value |
|--------|-------|
| Backbone parameters | ~2.54 M |
| Backbone FLOPs (256×256 patch) | ~0.06 GFLOPs |
| Model disk size | ~9.8 MB |
| Avg. inference latency (GPU) | < 25 ms / frame |
| Learned checkpoints | **None** |

---

## Repository Structure

```
repo/
├── README.md                   ← this file
├── Dockerfile                  ← fully reproducible inference pipeline
├── requirements.txt
├── train.py                    ← stub (no training needed — see note above)
├── track.py                    ← inference / prediction generation
├── models/
│   ├── __init__.py
│   └── feature_extractor.py   ← MobileNet-V3-Small wrapper (frozen weights)
├── configs/
│   └── tracker_config.yaml    ← all hyper-parameters
└── submission/
    ├── __init__.py
    ├── generate_csv.py         ← assembles final_submission.csv
    └── final_submission.csv   ← our Kaggle submission
```

---

## Environment Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

Or with Docker (GPU):

```bash
docker build -t aieees-tracker .
```

---

## Training

**No training required.** The MobileNet-V3-Small weights are loaded directly from
torchvision (ImageNet pretrained). Running `train.py` will print a reminder of this.

---

## Inference

### Option A — Python directly

```bash
python track.py \
    --dataset_root /path/to/contest_release \
    --save_root    results \
    --config       configs/tracker_config.yaml \
    --generate_csv
```

> The `--weights` argument is optional. When omitted, ImageNet pretrained weights
> are loaded automatically — no file download needed.

This writes per-sequence `predictions.txt` files under `results/` and
generates `submission/final_submission.csv`.

### Option B — Docker (GPU)

```bash
docker run --rm --gpus all \
    -v /path/to/data:/data \
    -v /path/to/results:/results \
    aieees-tracker \
    python track.py \
        --dataset_root /data/contest_release \
        --save_root    /results \
        --config       configs/tracker_config.yaml \
        --generate_csv
```

### Option C — Generate submission CSV only (from existing results)

```bash
python submission/generate_csv.py \
    --results_root results \
    --output       submission/final_submission.csv
```

---

## Output Format

`submission/final_submission.csv` follows the competition format:

```
id,x,y,w,h
dataset1/Car_video_0,536,551,226,142
dataset1/Car_video_1,536,552,226,142
...
```

Where the `id` field is `<dataset_name>/<sequence_name>_<frame_index>`.

---

## Configuration

All hyper-parameters live in `configs/tracker_config.yaml` and can be tuned
without touching the code. Key knobs:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `tracking_threshold` | 0.45 | CSRT confidence cut-off |
| `latency_budget_ms` | 25 | Skip search if frame is over-budget |
| `max_lost_before_freeze` | 15 | Switch to pure Kalman after N lost frames |
| `feature_update_interval` | 3 | Run MobileNet every N frames |
| `max_templates` | 8 | Template bank capacity |

---

## Technical Report

See `technical_report.pdf` in this repository.

---

## Contact

competition@mtc.edu.eg  ·  Team AIEEEs
