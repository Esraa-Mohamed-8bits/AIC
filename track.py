"""
track.py — AIC-4 Inference Script
===================================
Runs the AIEEEs tracker on one or more aerial tracking datasets and
produces per-sequence prediction files + a Kaggle-format submission CSV.

Usage
-----
Single dataset:
    python track.py --dataset_root /data/contest_release \
                    --save_root    /results \
                    --config       configs/tracker_config.yaml

Generate submission CSV afterwards:
    python submission/generate_csv.py \
        --results_root /results \
        --output       submission/final_submission.csv
"""

from __future__ import annotations

import argparse
import cv2
import glob
import json
import os
import time
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from models.feature_extractor import FeatureExtractor
from submission.generate_csv import build_submission_csv


# ============================================================
#  Load config
# ============================================================

def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ============================================================
#  Tracker sub-components
# ============================================================

class TemplateBank:
    """Maintains a diverse pool of appearance templates."""

    def __init__(self, initial_feature: torch.Tensor, max_templates: int,
                 diversity_threshold: float) -> None:
        self.templates          = [initial_feature]
        self.max_templates      = max_templates
        self.diversity_threshold = diversity_threshold
        self.original           = initial_feature

    def add(self, feature: torch.Tensor | None, score: float) -> None:
        if feature is None:
            return
        similarities = [
            F.cosine_similarity(t, feature, dim=0).item()
            for t in self.templates
        ]
        if max(similarities) < (1.0 - self.diversity_threshold):
            self.templates.append(feature)
            if len(self.templates) > self.max_templates:
                self.templates.pop(1)   # preserve the original at index 0

    def match(self, feature: torch.Tensor | None) -> float:
        if feature is None:
            return 0.0
        orig_sim = F.cosine_similarity(self.original, feature, dim=0).item()
        others   = [
            F.cosine_similarity(t, feature, dim=0).item()
            for t in self.templates[1:]
        ]
        return max(orig_sim * 1.1, max(others) if others else 0.0)


class ScaleController:
    """Clamps scale changes to a configurable factor."""

    def __init__(self, initial_box: tuple, max_change: float,
                 min_size: int, max_size: int) -> None:
        self.prev_box   = initial_box
        self.max_change = max_change
        self.min_size   = min_size
        self.max_size   = max_size

    def regulate(self, new_box: tuple) -> tuple:
        x, y, w, h       = new_box
        _, _, pw, ph     = self.prev_box
        w = max(pw / self.max_change, min(w, pw * self.max_change))
        h = max(ph / self.max_change, min(h, ph * self.max_change))
        w = max(self.min_size, min(w, self.max_size))
        h = max(self.min_size, min(h, self.max_size))
        regulated        = (int(x), int(y), int(w), int(h))
        self.prev_box    = regulated
        return regulated


class VelocityEstimator:
    """Estimates centre velocity over a rolling window."""

    def __init__(self, history_length: int) -> None:
        self.history = deque(maxlen=history_length)

    def update(self, box: tuple) -> None:
        self.history.append((box[0] + box[2] // 2, box[1] + box[3] // 2))

    def predict(self) -> tuple[float, float]:
        if len(self.history) < 2:
            return (0.0, 0.0)
        n  = len(self.history)
        vx = (self.history[-1][0] - self.history[0][0]) / n
        vy = (self.history[-1][1] - self.history[0][1]) / n
        return (vx, vy)


class OutOfViewManager:
    """Remembers which border the target exited from."""

    def __init__(self, border_zone: int, exit_memory: int) -> None:
        self.border_zone      = border_zone
        self.exit_memory      = exit_memory
        self.exit_direction   = None
        self.frames_since_exit = 0

    def check_exit(self, box: tuple, shape: tuple) -> None:
        H, W = shape[:2]
        x, y, w, h = box
        L = x < self.border_zone
        R = (x + w) > (W - self.border_zone)
        T = y < self.border_zone
        B = (y + h) > (H - self.border_zone)
        if any([L, R, T, B]):
            self.exit_direction    = {'left': L, 'right': R, 'top': T, 'bottom': B}
            self.frames_since_exit = 0
        elif self.exit_direction:
            self.frames_since_exit += 1
            if self.frames_since_exit > self.exit_memory:
                self.exit_direction = None

    def get_border_regions(self, shape: tuple, box_size: tuple) -> list[tuple]:
        if not self.exit_direction:
            return []
        H, W   = shape[:2]
        w, h   = box_size
        pos    = []
        step   = self.border_zone
        if self.exit_direction.get('left'):
            pos.extend([(0, y) for y in range(0, H - h, step)])
        if self.exit_direction.get('right'):
            pos.extend([(W - w, y) for y in range(0, H - h, step)])
        if self.exit_direction.get('top'):
            pos.extend([(x, 0) for x in range(0, W - w, step)])
        if self.exit_direction.get('bottom'):
            pos.extend([(x, H - h) for x in range(0, W - w, step)])
        return pos


# ============================================================
#  Helper functions
# ============================================================

def get_object_profile(init_box: tuple, frame_shape: tuple, cfg: dict) -> dict:
    H, W       = frame_shape[:2]
    area_ratio = (init_box[2] * init_box[3]) / (H * W)
    profiles   = cfg["size_profiles"]
    if area_ratio < profiles["tiny"]["area_max"]:
        return profiles["tiny"].copy()
    if area_ratio > profiles["large"]["area_min"]:
        return profiles["large"].copy()
    return profiles["medium"].copy()


def kalman_predict_box(box: tuple, velocity: tuple) -> tuple:
    vx, vy = velocity
    x, y, w, h = box
    return (int(x + vx), int(y + vy), w, h)


def get_dynamic_reinit_threshold(frames_lost: int, cfg: dict) -> float:
    thresholds = {
        int(k): v for k, v in cfg["dynamic_reinit_thresholds"].items()
    }
    result = cfg["reinit_threshold"]
    for lost_limit in sorted(thresholds):
        if frames_lost >= lost_limit:
            result = thresholds[lost_limit]
    return result


def grid_search(
    frame: np.ndarray,
    templates: TemplateBank,
    center_box: tuple,
    velocity: tuple,
    grid_size: int,
    motion_weight: float,
    extractor: FeatureExtractor,
    over_budget_fn,
    use_mobilenet: bool = True,
) -> tuple[float, tuple]:
    if not use_mobilenet:
        return 0.0, center_box

    H, W    = frame.shape[:2]
    x, y, w, h = center_box
    vx, vy  = velocity
    pred_x  = int(x + vx)
    pred_y  = int(y + vy)
    step_x  = max(1, w // 2)
    step_y  = max(1, h // 2)
    best_score, best_box = 0.0, center_box
    half    = grid_size // 2

    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            if over_budget_fn():
                return best_score, best_box
            sx = max(0, min(
                int(x * (1 - motion_weight) + pred_x * motion_weight) + i * step_x,
                W - w
            ))
            sy = max(0, min(
                int(y * (1 - motion_weight) + pred_y * motion_weight) + j * step_y,
                H - h
            ))
            feat  = extractor.extract(frame[sy:sy + h, sx:sx + w])
            score = templates.match(feat)
            if score > best_score:
                best_score, best_box = score, (sx, sy, w, h)

    return best_score, best_box


# ============================================================
#  Utility: locate frames / video
# ============================================================

def extract_frames_from_video(video_path: str, output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    cap       = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{frame_idx:06d}.jpg"), frame)
        frame_idx += 1
    cap.release()
    return frame_idx


def find_annotation_file(dataset_dir: str, sequence_name: str) -> str | None:
    base = sequence_name
    for suffix in ["_30", "_24", "_20", "_15"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    candidates = [
        os.path.join(dataset_dir, sequence_name, "annotation.txt"),
        os.path.join(dataset_dir, sequence_name, "annotations.txt"),
        os.path.join(dataset_dir, sequence_name, "groundtruth.txt"),
        os.path.join(dataset_dir, "annotation",  f"{sequence_name}.txt"),
        os.path.join(dataset_dir, "annotations", f"{sequence_name}.txt"),
        os.path.join(dataset_dir, "annotation",  f"{base}.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ============================================================
#  Metrics
# ============================================================

def compute_iou(box1: tuple, box2: tuple) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter    = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union    = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def compute_center_error(box1: tuple, box2: tuple) -> float:
    return float(np.sqrt(
        ((box1[0] + box1[2] / 2) - (box2[0] + box2[2] / 2)) ** 2 +
        ((box1[1] + box1[3] / 2) - (box2[1] + box2[3] / 2)) ** 2
    ))


def compute_sequence_metrics(predictions: list, ground_truth) -> dict:
    min_len     = min(len(predictions), len(ground_truth))
    preds       = predictions[:min_len]
    gts         = ground_truth[:min_len]
    ious        = [compute_iou(p, tuple(map(int, g[:4]))) for p, g in zip(preds, gts)]
    center_errs = [compute_center_error(p, tuple(map(int, g[:4]))) for p, g in zip(preds, gts)]
    return {
        "auc":           float(np.mean(ious)),
        "precision_20":  float(np.mean([1 if ce < 20 else 0 for ce in center_errs])),
        "avg_iou":       float(np.mean(ious)),
    }


# ============================================================
#  Per-sequence tracker
# ============================================================

def track_sequence(
    video_path: str,
    init_box: tuple,
    output_dir: str,
    sequence_name: str,
    extractor: FeatureExtractor,
    cfg: dict,
) -> dict | None:

    # --- Locate frames ---
    if video_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
        frames_dir   = os.path.join(output_dir, "frames")
        extract_frames_from_video(video_path, frames_dir)
        image_files  = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    else:
        image_files  = sorted(glob.glob(os.path.join(video_path, "*.jpg")))

    if not image_files:
        print(f"  [WARN] No frames found for {sequence_name}")
        return None

    # --- Initialise frame 0 ---
    frame0    = cv2.imread(image_files[0])
    init_feat = extractor.extract(
        frame0[init_box[1]: init_box[1] + init_box[3],
               init_box[0]: init_box[0] + init_box[2]]
    )

    # --- Object-size profile ---
    profile       = get_object_profile(init_box, frame0.shape, cfg)
    use_mobilenet = profile["use_mobilenet"]
    search_grid   = profile["grid"]
    throttle      = profile["throttle"]

    # --- Sub-components ---
    templates   = TemplateBank(init_feat, cfg["max_templates"], cfg["template_diversity"])
    scale_ctrl  = ScaleController(init_box, cfg["max_scale_change"],
                                  cfg["min_box_size"], cfg["max_box_size"])
    velocity    = VelocityEstimator(cfg["motion_history"])
    out_of_view = OutOfViewManager(cfg["border_zone"], cfg["exit_memory"])

    # --- CSRT tracker ---
    params = cv2.TrackerCSRT_Params()
    params.use_channel_weights = True
    params.admm_iterations     = 3
    tracker = cv2.TrackerCSRT_create(params)
    tracker.init(frame0, init_box)

    current_box           = init_box
    frames_lost           = 0
    predictions           = [init_box]
    latencies             = []
    feature_frame_counter = 0

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "tracking_log.csv")
    log_file = open(log_path, "w")
    log_file.write("frame,score,state,ms,x,y,w,h\n")

    for idx, img_path in enumerate(image_files[1:], start=1):
        t0    = time.time()
        frame = cv2.imread(img_path)

        over_budget = lambda: (time.time() - t0) * 1000 > cfg["latency_budget_ms"]

        # ---- FREEZE PATH ----
        if frames_lost > cfg["max_lost_before_freeze"]:
            current_box = kalman_predict_box(current_box, velocity.predict())
            H, W        = frame.shape[:2]
            x, y, w, h  = current_box
            current_box = (max(0, min(x, W - w)), max(0, min(y, H - h)), w, h)
            velocity.update(current_box)
            out_of_view.check_exit(current_box, frame.shape)
            dt = (time.time() - t0) * 1000
            latencies.append(dt)
            log_file.write(f"{idx},0.0000,FROZEN,{dt:.1f},{x},{y},{w},{h}\n")
            predictions.append(current_box)
            frames_lost += 1
            continue

        # ---- CSRT UPDATE ----
        success, csrt_box = tracker.update(frame)
        if success:
            csrt_box = scale_ctrl.regulate(tuple(map(int, csrt_box)))
            feature_frame_counter += 1
            if use_mobilenet and feature_frame_counter >= cfg["feature_update_interval"]:
                feature_frame_counter = 0
                csrt_feat  = extractor.extract(
                    frame[csrt_box[1]: csrt_box[1] + csrt_box[3],
                          csrt_box[0]: csrt_box[0] + csrt_box[2]]
                )
                csrt_score = templates.match(csrt_feat)
            else:
                csrt_score = cfg["tracking_threshold"] + 0.01
                csrt_feat  = None
        else:
            csrt_score = 0.0
            csrt_box   = current_box
            csrt_feat  = None

        # ---- LOST PATH ----
        if csrt_score < cfg["tracking_threshold"]:
            frames_lost   += 1
            should_search  = (frames_lost < 5) or (idx % throttle == 0)

            if should_search and not over_budget():
                gs = search_grid if frames_lost < 5 else min(search_grid + 2, 7)
                s_score, s_box = grid_search(
                    frame, templates, current_box,
                    velocity.predict(), gs,
                    cfg["motion_weight"],
                    extractor, over_budget,
                    use_mobilenet=use_mobilenet,
                )

                if use_mobilenet and not over_budget():
                    for bx, by in out_of_view.get_border_regions(
                            frame.shape, (current_box[2], current_box[3])):
                        if over_budget():
                            break
                        sc = templates.match(extractor.extract(
                            frame[by: by + current_box[3], bx: bx + current_box[2]]
                        ))
                        if sc > s_score:
                            s_score, s_box = sc, (bx, by, current_box[2], current_box[3])

                if s_score > csrt_score:
                    current_box   = s_box
                    final_score   = s_score
                    state         = "RECOVERED"
                    dyn_thresh    = get_dynamic_reinit_threshold(frames_lost, cfg)
                    if s_score > dyn_thresh:
                        tracker = cv2.TrackerCSRT_create(params)
                        tracker.init(frame, current_box)
                        frames_lost = 0
                else:
                    current_box = kalman_predict_box(current_box, velocity.predict())
                    final_score = csrt_score
                    state       = "LOST"
            else:
                current_box = kalman_predict_box(current_box, velocity.predict())
                final_score = csrt_score
                state       = "LOST_SKIP"

        # ---- TRACKING PATH ----
        else:
            current_box = csrt_box
            final_score = csrt_score
            frames_lost = 0
            state       = "TRACKING"
            if use_mobilenet and final_score > cfg["update_threshold"]:
                if csrt_feat is None:
                    csrt_feat = extractor.extract(
                        frame[current_box[1]: current_box[1] + current_box[3],
                              current_box[0]: current_box[0] + current_box[2]]
                    )
                templates.add(csrt_feat, final_score)

        # Clamp
        H, W        = frame.shape[:2]
        x, y, w, h  = current_box
        current_box = (max(0, min(x, W - w)), max(0, min(y, H - h)), w, h)

        velocity.update(current_box)
        out_of_view.check_exit(current_box, frame.shape)

        dt = (time.time() - t0) * 1000
        latencies.append(dt)
        x, y, w, h = current_box
        log_file.write(f"{idx},{final_score:.4f},{state},{dt:.1f},{x},{y},{w},{h}\n")
        predictions.append(current_box)

    log_file.close()

    return {
        "predictions":  predictions,
        "avg_latency":  float(np.mean(latencies)),
        "max_latency":  float(np.max(latencies)),
        "p95_latency":  float(np.percentile(latencies, 95)),
    }


# ============================================================
#  Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AIEEEs Tracker — AIC-4 Inference")
    p.add_argument("--dataset_root", required=True,
                   help="Root directory containing dataset folders.")
    p.add_argument("--save_root", default="results",
                   help="Directory to write per-sequence results to.")
    p.add_argument("--config", default="configs/tracker_config.yaml",
                   help="Path to tracker YAML config.")
    p.add_argument("--weights", default=None,
                   help="(Optional) Path to fine-tuned MobileNet weights.")
    p.add_argument("--generate_csv", action="store_true",
                   help="After inference, generate submission/final_submission.csv.")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    cfg       = load_config(args.config)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor(weights_path=args.weights, device=device)

    os.makedirs(args.save_root, exist_ok=True)
    all_results: list[dict] = []

    for dataset_name in sorted(os.listdir(args.dataset_root)):
        dataset_path = os.path.join(args.dataset_root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        print(f"\n{'='*60}\nProcessing {dataset_name}\n{'='*60}")

        for sequence_name in sorted(os.listdir(dataset_path)):
            sequence_path = os.path.join(dataset_path, sequence_name)
            if not os.path.isdir(sequence_path):
                continue
            print(f"\n  --- {sequence_name} ---")

            anno_file = find_annotation_file(dataset_path, sequence_name)
            if anno_file is None:
                print("    [WARN] No annotation file — skipping.")
                continue

            try:
                anno_data = np.genfromtxt(anno_file, delimiter=",")
                if np.isnan(anno_data).all():
                    anno_data = np.genfromtxt(anno_file)
                if anno_data.ndim == 1:
                    anno_data = anno_data.reshape(1, -1)
                init_box     = tuple(map(int, anno_data[0][:4]))
                has_full_gt  = (dataset_name != "dataset1" and len(anno_data) > 1)
            except Exception as exc:
                print(f"    [ERROR] reading annotation: {exc}")
                continue

            # Locate video or frame folder
            video_file = None
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                cands = glob.glob(os.path.join(sequence_path, f"*{ext}"))
                if cands:
                    video_file = cands[0]
                    break
            if video_file is None:
                frames = glob.glob(os.path.join(sequence_path, "*.jpg"))
                video_file = sequence_path if frames else None
            if video_file is None:
                print("    [WARN] No video/frames found — skipping.")
                continue

            output_dir = os.path.join(args.save_root, dataset_name, sequence_name)
            os.makedirs(output_dir, exist_ok=True)

            try:
                result = track_sequence(
                    video_file, init_box, output_dir, sequence_name, extractor, cfg
                )
                if result is None:
                    continue

                # Save predictions
                pred_file = os.path.join(output_dir, "predictions.txt")
                with open(pred_file, "w") as fh:
                    for box in result["predictions"]:
                        fh.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")

                row: dict = {
                    "dataset":         dataset_name,
                    "sequence":        sequence_name,
                    "avg_latency_ms":  result["avg_latency"],
                    "max_latency_ms":  result["max_latency"],
                    "p95_latency_ms":  result["p95_latency"],
                    "num_frames":      len(result["predictions"]),
                    "auc":             None,
                    "precision_20":    None,
                }

                if has_full_gt:
                    m = compute_sequence_metrics(result["predictions"], anno_data)
                    row.update({"auc": m["auc"], "precision_20": m["precision_20"]})
                    print(f"    AUC: {m['auc']:.4f}  Prec@20px: {m['precision_20']:.4f}  "
                          f"Avg latency: {result['avg_latency']:.1f}ms")
                else:
                    print(f"    Tracked {len(result['predictions'])} frames  "
                          f"| Avg latency: {result['avg_latency']:.1f}ms")

                all_results.append(row)

            except Exception as exc:
                import traceback
                print(f"    [ERROR] tracking: {exc}")
                traceback.print_exc()

    # ---- Summary ----
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.save_root, "all_results.csv"), index=False)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    df_gt = df[df["auc"].notna()]
    if len(df_gt) > 0:
        print(f"Mean AUC        : {df_gt['auc'].mean():.4f}")
        print(f"Mean Prec@20px  : {df_gt['precision_20'].mean():.4f}")
    print(f"Mean latency    : {df['avg_latency_ms'].mean():.2f} ms")
    print(f"Max  latency    : {df['max_latency_ms'].max():.2f} ms")
    print(f"P95  latency    : {df['p95_latency_ms'].quantile(0.95):.2f} ms")

    summary = {
        "total_sequences":        len(df),
        "overall_auc":            float(df_gt["auc"].mean()) if len(df_gt) else None,
        "overall_precision_20":   float(df_gt["precision_20"].mean()) if len(df_gt) else None,
        "overall_avg_latency_ms": float(df["avg_latency_ms"].mean()),
        "overall_max_latency_ms": float(df["max_latency_ms"].max()),
    }
    with open(os.path.join(args.save_root, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nResults saved to: {args.save_root}")

    if args.generate_csv:
        csv_path = os.path.join("submission", "final_submission.csv")
        n = build_submission_csv(args.save_root, csv_path)
        print(f"Submission CSV  : {csv_path}  ({n:,} rows)")


if __name__ == "__main__":
    main()
