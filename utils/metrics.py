"""
Evaluation metrics for small target detection.

Provides COCO-style AP breakdown by object size (small/medium/large),
specifically tailored for TT100K traffic sign detection.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# COCO size thresholds (in pixels, for absolute coordinates)
# small: area < 32^2, medium: 32^2 <= area < 96^2, large: area >= 96^2
SMALL_AREA_THRESH = 32 * 32    # 1024 px²
MEDIUM_AREA_THRESH = 96 * 96   # 9216 px²


def classify_by_area(bboxes: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Classify bounding boxes by area into small/medium/large.

    Args:
        bboxes: (N, 4) array in [x1, y1, x2, y2] absolute coordinates.

    Returns:
        Dict with keys 'small', 'medium', 'large', each containing
        a boolean mask array of shape (N,).
    """
    if len(bboxes) == 0:
        return {
            "small": np.array([], dtype=bool),
            "medium": np.array([], dtype=bool),
            "large": np.array([], dtype=bool),
        }

    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    areas = widths * heights

    return {
        "small": areas < SMALL_AREA_THRESH,
        "medium": (areas >= SMALL_AREA_THRESH) & (areas < MEDIUM_AREA_THRESH),
        "large": areas >= MEDIUM_AREA_THRESH,
    }


def compute_size_breakdown(
    predictions: List[np.ndarray],
    image_size: Tuple[int, int],
) -> Dict[str, int]:
    """
    Compute the count of predictions by object size.

    Args:
        predictions: List of (M, 6) arrays [x1, y1, x2, y2, confidence, class].
        image_size: (height, width) of the original image.

    Returns:
        Dict with counts for 'small', 'medium', 'large'.
    """
    counts = {"small": 0, "medium": 0, "large": 0}
    for pred in predictions:
        if len(pred) == 0:
            continue
        masks = classify_by_area(pred[:, :4])
        for size in counts:
            counts[size] += masks[size].sum()
    return counts


def compute_small_target_ap(
    coco_eval_results,
    area_rng: str = "small",
) -> float:
    """
    Extract AP for a specific area range from COCO eval results.

    Args:
        coco_eval_results: Results from coco_eval.evaluate() and accumulate()
                          or from ultralytics validation.
        area_rng: One of 'small', 'medium', 'large', 'all'.

    Returns:
        AP value (float).
    """
    area_indices = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
    }
    idx = area_indices.get(area_rng, 0)

    # Handle ultralytics results format
    if hasattr(coco_eval_results, "stats"):
        stats = coco_eval_results.stats
        if len(stats) > idx:
            return float(stats[idx])
    # Handle dict format
    elif isinstance(coco_eval_results, dict):
        return float(coco_eval_results.get(f"AP_{area_rng}", 0.0))

    return 0.0


def format_metrics_table(metrics: Dict) -> str:
    """
    Format evaluation metrics as a readable table.

    Args:
        metrics: Dictionary of metric values.

    Returns:
        Formatted string.
    """
    rows = []
    rows.append("=" * 50)
    rows.append("Evaluation Metrics")
    rows.append("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            rows.append(f"  {k:<25s}: {v:.4f}")
        else:
            rows.append(f"  {k:<25s}: {v}")
    rows.append("=" * 50)
    return "\n".join(rows)


def save_metrics(metrics: Dict, filepath: Path) -> None:
    """
    Save metrics dictionary to a JSON file.

    Args:
        metrics: Dictionary of metric values.
        filepath: Path to output JSON file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types to native Python types
    cleaned = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            cleaned[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, np.ndarray):
            cleaned[k] = v.tolist()
        else:
            cleaned[k] = v
    with open(filepath, "w") as f:
        json.dump(cleaned, f, indent=2)


def load_metrics(filepath: Path) -> Dict:
    """Load metrics from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def analyze_dataset_objects(
    label_dir: Path, class_names: List[str]
) -> Dict:
    """
    Analyze labeled objects in a YOLO-format dataset.

    Computes per-class counts and size distribution (small/medium/large)
    to understand the dataset composition before training.

    Args:
        label_dir: Directory containing YOLO-format .txt label files.
        class_names: List of class name strings.

    Returns:
        Dict with keys: total_boxes, per_class, size_distribution.
    """
    total = 0
    per_class = {name: 0 for name in class_names}
    size_dist = {"small": 0, "medium": 0, "large": 0}

    for label_file in label_dir.glob("*.txt"):
        if not label_file.is_file():
            continue
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # YOLO format is normalized; we report normalized sizes
                # For actual area in pixels, multiply by image dimensions
                # Here we use normalized area as a proxy
                area_norm = w_norm * h_norm

                total += 1
                if cls_id < len(class_names):
                    per_class[class_names[cls_id]] += 1

                if area_norm < 0.001:     # ~32² on 1280² image
                    size_dist["small"] += 1
                elif area_norm < 0.01:     # ~96² on 1280² image
                    size_dist["medium"] += 1
                else:
                    size_dist["large"] += 1

    return {
        "total_boxes": total,
        "per_class": per_class,
        "size_distribution": size_dist,
    }
