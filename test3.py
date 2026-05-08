import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import os
import glob
import time
from collections import deque
import pandas as pd
import json

# ================= CONFIG =================
DATASET_ROOT = r"D:\AIC4\data\contest_release"
SAVE_ROOT = r"D:\AIC4\results_v4_7_full_test"
os.makedirs(SAVE_ROOT, exist_ok=True)

CONFIG = {
    # Thresholds
    'tracking_threshold': 0.45,
    'reinit_threshold': 0.60,
    'update_threshold': 0.92,

    # Template bank
    'max_templates': 8,
    'template_diversity': 0.12,

    # Scale control
    'max_scale_change': 1.08,
    'min_box_size': 15,
    'max_box_size': 900,

    # Motion
    'motion_history': 6,
    'motion_weight': 0.75,

    # Search grid
    'grid_sizes': {'tracking': 3, 'searching': 5, 'desperate': 7},
    'search_throttle': 3,

    # Out-of-view
    'border_zone': 60,
    'exit_memory': 40,

    # Feature extraction
    'resize_dim': (256, 256),

    # --- NEW v4.7 ---
    # Hard latency budget per frame (ms). If elapsed > this before grid search, skip search.
    'latency_budget_ms': 25,

    # Max consecutive lost frames before freezing (no search, just Kalman propagation)
    'max_lost_before_freeze': 15,

    # Feature extraction frequency during normal tracking (call MobileNet every N frames)
    'feature_update_interval': 3,

    # Dynamic reinit thresholds based on frames_lost
    'dynamic_reinit_thresholds': {
        20: 0.40,
        50: 0.30,
    },
}

# Object-size profile thresholds (fraction of frame area)
SIZE_PROFILES = {
    'tiny':  {'area_max': 0.005, 'grid': 3, 'throttle': 2, 'use_mobilenet': False},
    'large': {'area_min': 0.10,  'grid': 3, 'throttle': 5, 'use_mobilenet': False},
    'medium':{'grid': 5,         'throttle': 3, 'use_mobilenet': True},
}

# ================= FEATURE EXTRACTOR (RGB) =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
base_model.classifier = nn.Identity()
model = base_model.to(device).eval()

preprocess = T.Compose([
    T.Resize(CONFIG['resize_dim']),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(patch_bgr):
    """Extract normalized features from an RGB patch."""
    if patch_bgr is None or patch_bgr.shape[0] < 5 or patch_bgr.shape[1] < 5:
        return None
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(patch_rgb)
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return F.normalize(model(tensor).squeeze(0), p=2, dim=0)

# ================= COMPONENTS =================

class TemplateBank:
    def __init__(self, initial_feature, max_templates, diversity_threshold):
        self.templates = [initial_feature]
        self.max_templates = max_templates
        self.diversity_threshold = diversity_threshold
        self.original = initial_feature

    def add(self, feature, score):
        if feature is None:
            return
        similarities = [F.cosine_similarity(t, feature, dim=0).item() for t in self.templates]
        if max(similarities) < (1.0 - self.diversity_threshold):
            self.templates.append(feature)
            if len(self.templates) > self.max_templates:
                self.templates.pop(1)  # preserve original at index 0

    def match(self, feature):
        if feature is None:
            return 0.0
        orig_sim = F.cosine_similarity(self.original, feature, dim=0).item()
        others = [F.cosine_similarity(t, feature, dim=0).item() for t in self.templates[1:]]
        return max(orig_sim * 1.1, max(others) if others else 0.0)


class ScaleController:
    def __init__(self, initial_box, max_change, min_size, max_size):
        self.prev_box = initial_box
        self.max_change = max_change
        self.min_size = min_size
        self.max_size = max_size

    def regulate(self, new_box):
        x, y, w, h = new_box
        px, py, pw, ph = self.prev_box
        w = max(pw / self.max_change, min(w, pw * self.max_change))
        h = max(ph / self.max_change, min(h, ph * self.max_change))
        w = max(self.min_size, min(w, self.max_size))
        h = max(self.min_size, min(h, self.max_size))
        regulated = (int(x), int(y), int(w), int(h))
        self.prev_box = regulated
        return regulated


class VelocityEstimator:
    def __init__(self, history_length):
        self.history = deque(maxlen=history_length)

    def update(self, box):
        self.history.append((box[0] + box[2] // 2, box[1] + box[3] // 2))

    def predict(self):
        if len(self.history) < 2:
            return (0, 0)
        n = len(self.history)
        vx = (self.history[-1][0] - self.history[0][0]) / n
        vy = (self.history[-1][1] - self.history[0][1]) / n
        return (vx, vy)


class OutOfViewManager:
    def __init__(self, border_zone, exit_memory):
        self.border_zone = border_zone
        self.exit_memory = exit_memory
        self.exit_direction = None
        self.frames_since_exit = 0

    def check_exit(self, box, shape):
        H, W = shape[:2]
        x, y, w, h = box
        L = x < self.border_zone
        R = (x + w) > (W - self.border_zone)
        T = y < self.border_zone
        B = (y + h) > (H - self.border_zone)
        if any([L, R, T, B]):
            self.exit_direction = {'left': L, 'right': R, 'top': T, 'bottom': B}
            self.frames_since_exit = 0
        elif self.exit_direction:
            self.frames_since_exit += 1
            if self.frames_since_exit > self.exit_memory:
                self.exit_direction = None

    def get_border_regions(self, shape, box_size):
        if not self.exit_direction:
            return []
        H, W = shape[:2]
        w, h = box_size
        pos = []
        step = self.border_zone
        if self.exit_direction.get('left'):
            pos.extend([(0, y) for y in range(0, H - h, step)])
        if self.exit_direction.get('right'):
            pos.extend([(W - w, y) for y in range(0, H - h, step)])
        if self.exit_direction.get('top'):
            pos.extend([(x, 0) for x in range(0, W - w, step)])
        if self.exit_direction.get('bottom'):
            pos.extend([(x, H - h) for x in range(0, W - w, step)])
        return pos


# ================= HELPER FUNCTIONS =================

def get_object_profile(init_box, frame_shape):
    """
    Classify the tracked object into tiny / large / medium based on
    its bounding-box area relative to the frame, and return a config
    profile that controls grid size, search throttle, and MobileNet usage.
    """
    H, W = frame_shape[:2]
    area_ratio = (init_box[2] * init_box[3]) / (H * W)
    if area_ratio < SIZE_PROFILES['tiny']['area_max']:
        return SIZE_PROFILES['tiny'].copy()
    if area_ratio > SIZE_PROFILES['large']['area_min']:
        return SIZE_PROFILES['large'].copy()
    return SIZE_PROFILES['medium'].copy()


def kalman_predict_box(box, velocity):
    """Advance the bounding box by one step using the current velocity estimate."""
    vx, vy = velocity
    x, y, w, h = box
    return (int(x + vx), int(y + vy), w, h)


def get_dynamic_reinit_threshold(frames_lost):
    """
    Lower the re-init threshold progressively the longer we have been lost,
    so we become more aggressive about restarting the CSRT tracker.
    """
    thresholds = CONFIG['dynamic_reinit_thresholds']
    result = CONFIG['reinit_threshold']
    for lost_limit in sorted(thresholds.keys()):
        if frames_lost >= lost_limit:
            result = thresholds[lost_limit]
    return result


def grid_search(frame, templates, center_box, velocity, grid_size,
                motion_weight, time_budget_fn, use_mobilenet=True):
    """
    Scan a grid of candidate positions around the predicted location.
    Respects the per-frame time budget: stops early if budget is exceeded.
    When use_mobilenet is False the function returns immediately with score 0,
    so the caller can fall back to Kalman propagation.
    """
    if not use_mobilenet:
        return 0.0, center_box

    H, W = frame.shape[:2]
    x, y, w, h = center_box
    vx, vy = velocity
    pred_x = int(x + vx)
    pred_y = int(y + vy)
    step_x = max(1, w // 2)
    step_y = max(1, h // 2)
    best_score, best_box = 0.0, center_box
    half = grid_size // 2

    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            # Abort if we are running over the latency budget
            if time_budget_fn():
                return best_score, best_box

            sx = max(0, min(int(x * (1 - motion_weight) + pred_x * motion_weight) + i * step_x, W - w))
            sy = max(0, min(int(y * (1 - motion_weight) + pred_y * motion_weight) + j * step_y, H - h))
            feat = extract_features(frame[sy:sy + h, sx:sx + w])
            score = templates.match(feat)
            if score > best_score:
                best_score, best_box = score, (sx, sy, w, h)

    return best_score, best_box


# ================= METRICS COMPUTATION =================

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_center_error(box1, box2):
    c1_x = box1[0] + box1[2] / 2
    c1_y = box1[1] + box1[3] / 2
    c2_x = box2[0] + box2[2] / 2
    c2_y = box2[1] + box2[3] / 2
    return np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)


def compute_sequence_metrics(predictions, ground_truth):
    if len(predictions) != len(ground_truth):
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]

    ious, center_errors = [], []
    for pred, gt in zip(predictions, ground_truth):
        ious.append(compute_iou(pred, gt))
        center_errors.append(compute_center_error(pred, gt))

    auc = np.mean(ious)
    precision_20 = np.mean([1 if ce < 20 else 0 for ce in center_errors])
    return {
        'auc': auc,
        'precision_20': precision_20,
        'avg_iou': auc,
        'ious': ious,
        'center_errors': center_errors,
    }


# ================= VIDEO PROCESSING =================

def extract_frames_from_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"{frame_idx:06d}.jpg"), frame)
        frame_idx += 1
    cap.release()
    return frame_idx


def find_annotation_file(dataset_dir, sequence_name):
    base_name = sequence_name
    for suffix in ['_30', '_24', '_20', '_15']:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break

    possible_paths = [
        os.path.join(dataset_dir, sequence_name, "annotation.txt"),
        os.path.join(dataset_dir, sequence_name, "annotations.txt"),
        os.path.join(dataset_dir, sequence_name, "groundtruth.txt"),
        os.path.join(dataset_dir, "annotation", f"{sequence_name}.txt"),
        os.path.join(dataset_dir, "annotations", f"{sequence_name}.txt"),
        os.path.join(dataset_dir, "annotation", f"{base_name}.txt"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


# ================= TRACKER =================

def track_sequence(video_path, init_box, output_dir, sequence_name):
    """Track a single sequence and return predictions + latency stats."""

    # ---- Locate frames ----
    if video_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        frames_dir = os.path.join(output_dir, "frames")
        extract_frames_from_video(video_path, frames_dir)
        image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    else:
        image_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))

    if not image_files:
        print(f"  No frames found for {sequence_name}")
        return None

    # ---- Initialise frame 0 ----
    frame0 = cv2.imread(image_files[0])
    init_feat = extract_features(
        frame0[init_box[1]:init_box[1] + init_box[3],
               init_box[0]:init_box[0] + init_box[2]]
    )

    # ---- Object-size profile (computed once at init) ----
    profile = get_object_profile(init_box, frame0.shape)
    use_mobilenet = profile['use_mobilenet']
    search_grid   = profile['grid']
    throttle      = profile['throttle']

    # ---- Sub-components ----
    templates  = TemplateBank(init_feat, CONFIG['max_templates'], CONFIG['template_diversity'])
    scale_ctrl = ScaleController(init_box, CONFIG['max_scale_change'],
                                 CONFIG['min_box_size'], CONFIG['max_box_size'])
    velocity   = VelocityEstimator(CONFIG['motion_history'])
    out_of_view = OutOfViewManager(CONFIG['border_zone'], CONFIG['exit_memory'])

    # ---- CSRT tracker with reduced admm iterations ----
    params = cv2.TrackerCSRT_Params()
    params.use_channel_weights = True
    params.admm_iterations = 3          # default is 4; saves ~10% CSRT time
    tracker = cv2.TrackerCSRT_create(params)
    tracker.init(frame0, init_box)

    current_box  = init_box
    frames_lost  = 0
    predictions  = [init_box]
    latencies    = []

    # Frame-level feature-update counter (MobileNet throttle during normal tracking)
    feature_frame_counter = 0

    log_file = open(os.path.join(output_dir, "tracking_log.csv"), "w")
    log_file.write("frame,score,state,ms,x,y,w,h\n")

    for idx, img_path in enumerate(image_files[1:], start=1):
        t0 = time.time()
        frame = cv2.imread(img_path)

        # Helper: returns True when the per-frame latency budget is exhausted.
        over_budget = lambda: (time.time() - t0) * 1000 > CONFIG['latency_budget_ms']

        # ------------------------------------------------------------------ #
        # FREEZE PATH: object has been lost for too long → pure Kalman,       #
        # no search, no MobileNet.                                             #
        # ------------------------------------------------------------------ #
        if frames_lost > CONFIG['max_lost_before_freeze']:
            current_box = kalman_predict_box(current_box, velocity.predict())
            # Clamp to frame boundaries
            H, W = frame.shape[:2]
            x, y, w, h = current_box
            x = max(0, min(x, W - w))
            y = max(0, min(y, H - h))
            current_box = (x, y, w, h)

            velocity.update(current_box)
            out_of_view.check_exit(current_box, frame.shape)
            dt = (time.time() - t0) * 1000
            latencies.append(dt)
            log_file.write(f"{idx},0.0000,FROZEN,{dt:.1f},{x},{y},{w},{h}\n")
            predictions.append(current_box)
            frames_lost += 1
            continue

        # ------------------------------------------------------------------ #
        # NORMAL CSRT UPDATE                                                  #
        # ------------------------------------------------------------------ #
        success, csrt_box = tracker.update(frame)
        if success:
            csrt_box = scale_ctrl.regulate(tuple(map(int, csrt_box)))

            # Only call MobileNet every N frames during steady tracking to save time.
            feature_frame_counter += 1
            if use_mobilenet and feature_frame_counter >= CONFIG['feature_update_interval']:
                feature_frame_counter = 0
                csrt_feat  = extract_features(
                    frame[csrt_box[1]:csrt_box[1] + csrt_box[3],
                          csrt_box[0]:csrt_box[0] + csrt_box[2]]
                )
                csrt_score = templates.match(csrt_feat)
            else:
                # Skip MobileNet this frame; use a neutral score to let CSRT lead.
                csrt_score = CONFIG['tracking_threshold'] + 0.01
                csrt_feat  = None
        else:
            csrt_score = 0.0
            csrt_box   = current_box
            csrt_feat  = None

        # ------------------------------------------------------------------ #
        # LOST PATH: CSRT score below threshold                               #
        # ------------------------------------------------------------------ #
        if csrt_score < CONFIG['tracking_threshold']:
            frames_lost += 1

            # Decide whether to run search this frame (throttle)
            should_search = (frames_lost < 5) or (idx % throttle == 0)

            if should_search and not over_budget():
                # Determine grid size from profile
                gs = search_grid if frames_lost < 5 else min(search_grid + 2, 7)

                s_score, s_box = grid_search(
                    frame, templates, current_box,
                    velocity.predict(), gs,
                    CONFIG['motion_weight'],
                    over_budget,
                    use_mobilenet=use_mobilenet,
                )

                # Border-zone sweep (only if MobileNet is enabled and still in budget)
                if use_mobilenet and not over_budget():
                    for bx, by in out_of_view.get_border_regions(
                            frame.shape, (current_box[2], current_box[3])):
                        if over_budget():
                            break
                        sc = templates.match(extract_features(
                            frame[by:by + current_box[3], bx:bx + current_box[2]]
                        ))
                        if sc > s_score:
                            s_score, s_box = sc, (bx, by, current_box[2], current_box[3])

                if s_score > csrt_score:
                    current_box  = s_box
                    final_score  = s_score
                    state        = "RECOVERED"

                    # Dynamic reinit threshold: lower it the longer we've been lost
                    dynamic_thresh = get_dynamic_reinit_threshold(frames_lost)
                    if s_score > dynamic_thresh:
                        tracker = cv2.TrackerCSRT_create(params)
                        tracker.init(frame, current_box)
                        frames_lost = 0
                else:
                    current_box = kalman_predict_box(current_box, velocity.predict())
                    final_score = csrt_score
                    state       = "LOST"
            else:
                # Over budget or skipping this frame: pure Kalman propagation
                current_box = kalman_predict_box(current_box, velocity.predict())
                final_score = csrt_score
                state       = "LOST_SKIP"

        # ------------------------------------------------------------------ #
        # TRACKING PATH: CSRT is confident                                   #
        # ------------------------------------------------------------------ #
        else:
            current_box  = csrt_box
            final_score  = csrt_score
            frames_lost  = 0
            state        = "TRACKING"

            # Update template bank only when score is high AND MobileNet is enabled
            if use_mobilenet and final_score > CONFIG['update_threshold']:
                if csrt_feat is None:
                    csrt_feat = extract_features(
                        frame[current_box[1]:current_box[1] + current_box[3],
                              current_box[0]:current_box[0] + current_box[2]]
                    )
                templates.add(csrt_feat, final_score)

        # Clamp box to frame boundaries
        H, W = frame.shape[:2]
        x, y, w, h = current_box
        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))
        current_box = (x, y, w, h)

        velocity.update(current_box)
        out_of_view.check_exit(current_box, frame.shape)

        dt = (time.time() - t0) * 1000
        latencies.append(dt)
        x, y, w, h = current_box
        log_file.write(f"{idx},{final_score:.4f},{state},{dt:.1f},{x},{y},{w},{h}\n")
        predictions.append(current_box)

    log_file.close()

    return {
        'predictions': predictions,
        'latencies': latencies,
        'avg_latency': np.mean(latencies),
        'max_latency': np.max(latencies),
        'p95_latency': np.percentile(latencies, 95),
    }


# ================= MAIN =================

def main():
    all_results = []

    for dataset_name in sorted(os.listdir(DATASET_ROOT)):
        dataset_path = os.path.join(DATASET_ROOT, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print(f"{'='*60}")

        for sequence_name in sorted(os.listdir(dataset_path)):
            sequence_path = os.path.join(dataset_path, sequence_name)
            if not os.path.isdir(sequence_path):
                continue

            print(f"\n--- {sequence_name} ---")

            anno_file = find_annotation_file(dataset_path, sequence_name)
            if anno_file is None:
                print(f"  ⚠ No annotation file found, skipping")
                continue

            try:
                anno_data = np.genfromtxt(anno_file, delimiter=',')
                if np.isnan(anno_data).all():
                    anno_data = np.genfromtxt(anno_file)
                if anno_data.ndim == 1:
                    anno_data = anno_data.reshape(1, -1)

                init_box = tuple(map(int, anno_data[0][:4]))
                has_full_gt = (dataset_name != "dataset1" and len(anno_data) > 1)

            except Exception as e:
                print(f"  ⚠ Error reading annotation: {e}")
                continue

            # Locate video or frame directory
            video_file = None
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                candidates = glob.glob(os.path.join(sequence_path, f"*{ext}"))
                if candidates:
                    video_file = candidates[0]
                    break

            if video_file is None:
                frames = glob.glob(os.path.join(sequence_path, "*.jpg"))
                if frames:
                    video_file = sequence_path
                else:
                    print(f"  ⚠ No video or frames found")
                    continue

            output_dir = os.path.join(SAVE_ROOT, dataset_name, sequence_name)
            os.makedirs(output_dir, exist_ok=True)

            try:
                result = track_sequence(video_file, init_box, output_dir, sequence_name)
                if result is None:
                    continue

                # Save predictions
                pred_file = os.path.join(output_dir, "predictions.txt")
                with open(pred_file, 'w') as f:
                    for box in result['predictions']:
                        f.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")

                if has_full_gt:
                    metrics = compute_sequence_metrics(result['predictions'], anno_data)
                    seq_result = {
                        'dataset': dataset_name,
                        'sequence': sequence_name,
                        'auc': metrics['auc'],
                        'precision_20': metrics['precision_20'],
                        'avg_iou': metrics['avg_iou'],
                        'avg_latency_ms': result['avg_latency'],
                        'max_latency_ms': result['max_latency'],
                        'p95_latency_ms': result['p95_latency'],
                        'num_frames': len(result['predictions']),
                    }
                    print(f"  AUC: {metrics['auc']:.4f}")
                    print(f"  Precision@20px: {metrics['precision_20']:.4f}")
                    print(f"  Avg Latency: {result['avg_latency']:.2f}ms")
                    print(f"  Max Latency: {result['max_latency']:.2f}ms")
                else:
                    seq_result = {
                        'dataset': dataset_name,
                        'sequence': sequence_name,
                        'auc': None,
                        'precision_20': None,
                        'avg_iou': None,
                        'avg_latency_ms': result['avg_latency'],
                        'max_latency_ms': result['max_latency'],
                        'p95_latency_ms': result['p95_latency'],
                        'num_frames': len(result['predictions']),
                    }
                    print(f"  Tracked {len(result['predictions'])} frames")
                    print(f"  Avg Latency: {result['avg_latency']:.2f}ms")

                all_results.append(seq_result)

            except Exception as e:
                print(f"  ✗ Error tracking: {e}")
                import traceback
                traceback.print_exc()
                continue

    # ---- Aggregate results ----
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(SAVE_ROOT, "all_results.csv"), index=False)

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    sequences_with_gt = df[df['auc'].notna()]
    if len(sequences_with_gt) > 0:
        print(f"\nAccuracy Metrics ({len(sequences_with_gt)} sequences):")
        print(f"  Mean AUC:           {sequences_with_gt['auc'].mean():.4f} ± {sequences_with_gt['auc'].std():.4f}")
        print(f"  Mean Precision@20px:{sequences_with_gt['precision_20'].mean():.4f} ± {sequences_with_gt['precision_20'].std():.4f}")
        print(f"  Mean IoU:           {sequences_with_gt['avg_iou'].mean():.4f} ± {sequences_with_gt['avg_iou'].std():.4f}")

    print(f"\nEfficiency Metrics ({len(df)} sequences):")
    print(f"  Mean Latency: {df['avg_latency_ms'].mean():.2f}ms ± {df['avg_latency_ms'].std():.2f}ms")
    print(f"  Max Latency:  {df['max_latency_ms'].max():.2f}ms")
    print(f"  P95 Latency:  {df['p95_latency_ms'].quantile(0.95):.2f}ms")

    print(f"\nPer-Dataset Breakdown:")
    for dataset in sorted(df['dataset'].unique()):
        ddf = df[df['dataset'] == dataset]
        dwgt = ddf[ddf['auc'].notna()]
        print(f"\n  {dataset} ({len(ddf)} sequences):")
        if len(dwgt) > 0:
            print(f"    AUC:             {dwgt['auc'].mean():.4f}")
            print(f"    Precision@20px:  {dwgt['precision_20'].mean():.4f}")
        print(f"    Avg Latency:     {ddf['avg_latency_ms'].mean():.2f}ms")

    summary = {
        'total_sequences': len(df),
        'sequences_with_gt': len(sequences_with_gt),
        'overall_auc': float(sequences_with_gt['auc'].mean()) if len(sequences_with_gt) > 0 else None,
        'overall_precision_20': float(sequences_with_gt['precision_20'].mean()) if len(sequences_with_gt) > 0 else None,
        'overall_avg_latency_ms': float(df['avg_latency_ms'].mean()),
        'overall_max_latency_ms': float(df['max_latency_ms'].max()),
        'per_dataset': {},
    }
    for dataset in sorted(df['dataset'].unique()):
        ddf = df[df['dataset'] == dataset]
        dwgt = ddf[ddf['auc'].notna()]
        summary['per_dataset'][dataset] = {
            'num_sequences': len(ddf),
            'auc': float(dwgt['auc'].mean()) if len(dwgt) > 0 else None,
            'precision_20': float(dwgt['precision_20'].mean()) if len(dwgt) > 0 else None,
            'avg_latency_ms': float(ddf['avg_latency_ms'].mean()),
        }

    with open(os.path.join(SAVE_ROOT, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {SAVE_ROOT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()