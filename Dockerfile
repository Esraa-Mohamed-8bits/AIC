# ============================================================
#  AIEEEs Tracker — Dockerfile
#  MTC-AIC4 Phase I
# ============================================================
#  Build:
#      docker build -t aieees-tracker .
#
#  Run inference (no checkpoint needed — ImageNet weights auto-downloaded):
#      docker run --rm --gpus all \
#          -v /path/to/data:/data \
#          -v /path/to/results:/results \
#          aieees-tracker \
#          python track.py \
#              --dataset_root /data/contest_release \
#              --save_root    /results \
#              --config       configs/tracker_config.yaml \
#              --generate_csv
# ============================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application code ----
COPY configs/    configs/
COPY models/     models/
COPY submission/ submission/
COPY track.py    track.py
COPY train.py    train.py

# Pre-download MobileNet-V3-Small ImageNet weights so inference works offline
# No task-specific checkpoint is needed — this is the only weights file required.
RUN python - <<'EOF'
import torchvision.models as m
m.mobilenet_v3_small(weights=m.MobileNet_V3_Small_Weights.DEFAULT)
print("MobileNet-V3-Small ImageNet weights cached.")
EOF

# Default command: run inference and generate CSV (no --weights flag needed)
CMD ["python", "track.py", \
     "--dataset_root", "/data/contest_release", \
     "--save_root",    "/results", \
     "--config",       "configs/tracker_config.yaml", \
     "--generate_csv"]
