"""
submission/generate_csv.py
---------------------------
Collects per-sequence prediction files produced by track.py and assembles
them into the final Kaggle-format CSV:

    id,x,y,w,h
    dataset1/sequence_name_0,x0,y0,w0,h0
    dataset1/sequence_name_1,x1,y1,w1,h1
    ...

Usage (standalone):
    python submission/generate_csv.py \
        --results_root /path/to/results \
        --output       /path/to/final_submission.csv
"""

from __future__ import annotations

import argparse
import os
import glob
import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def collect_predictions(results_root: str) -> list[dict]:
    """
    Walk the results directory tree and collect every prediction.

    Directory layout expected (produced by track.py):
        results_root/
            <dataset_name>/
                <sequence_name>/
                    predictions.txt   ← one "x,y,w,h" line per frame

    Returns a list of dicts, one per (sequence, frame):
        [{"id": "dataset1/seq_name_<frame_idx>",
          "x": int, "y": int, "w": int, "h": int}, ...]
    """
    rows: list[dict] = []

    for pred_file in sorted(glob.glob(
            os.path.join(results_root, "**", "predictions.txt"), recursive=True)):

        pred_path = Path(pred_file)
        # e.g. results_root / dataset1 / Car_video / predictions.txt
        sequence_name = pred_path.parent.name
        dataset_name  = pred_path.parent.parent.name

        with open(pred_file, "r") as fh:
            for frame_idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    parts = line.split()           # fallback: whitespace-separated
                x, y, w, h = int(float(parts[0])), int(float(parts[1])), \
                              int(float(parts[2])), int(float(parts[3]))
                rows.append({
                    "id": f"{dataset_name}/{sequence_name}_{frame_idx}",
                    "x":  x,
                    "y":  y,
                    "w":  w,
                    "h":  h,
                })

    return rows


def build_submission_csv(results_root: str, output_path: str) -> int:
    """
    Build the final submission CSV and write it to *output_path*.
    Returns the number of rows written.
    """
    rows = collect_predictions(results_root)

    if not rows:
        raise RuntimeError(
            f"No predictions found under '{results_root}'. "
            "Make sure track.py has been run first."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "x", "y", "w", "h"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assemble per-sequence predictions into a Kaggle submission CSV."
    )
    p.add_argument(
        "--results_root", required=True,
        help="Root directory of tracking results (output of track.py)."
    )
    p.add_argument(
        "--output", default="submission/final_submission.csv",
        help="Path for the output CSV file."
    )
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    n_rows = build_submission_csv(args.results_root, args.output)
    print(f"✓ Wrote {n_rows:,} rows → {args.output}")
