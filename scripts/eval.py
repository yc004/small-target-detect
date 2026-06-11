#!/usr/bin/env python3
"""
Evaluation script for small target traffic sign detection.

Computes COCO-style metrics with a focus on small target performance:
- mAP@0.5, mAP@0.5:0.95 (per-class and overall)
- AP by object size: AP_S (small), AP_M (medium), AP_L (large)
- Precision-Recall curves
- Per-class breakdown

Usage:
    # Evaluate on test set
    python scripts/eval.py --weights runs/stage1_baseline/weights/best.pt \\
                           --data data/processed/dataset.yaml \\
                           --split test

    # With size breakdown analysis
    python scripts/eval.py --weights best.pt --data dataset.yaml --analyze_sizes
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import (
    SMALL_AREA_THRESH,
    MEDIUM_AREA_THRESH,
    classify_by_area,
    format_metrics_table,
    save_metrics,
)


def run_evaluation(
    weights_path: Path,
    data_yaml: Path,
    split: str = "test",
    imgsz: int = 1280,
    batch: int = 8,
    device: str = "0",
    conf: float = 0.001,
    iou: float = 0.7,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Run evaluation using ultralytics val mode.

    Args:
        weights_path: Path to trained model weights.
        data_yaml: Path to dataset.yaml.
        split: Dataset split to evaluate ('test' or 'val').
        imgsz: Image size.
        batch: Batch size.
        device: GPU device.
        conf: Confidence threshold.
        iou: IoU threshold for NMS.
        save_dir: Directory to save evaluation outputs.

    Returns:
        Dictionary of metrics.
    """
    from ultralytics import YOLO

    logger.info(f"Loading model from {weights_path}")
    model = YOLO(str(weights_path))

    # Temporarily modify data_yaml to point to test split if needed
    if split == "test":
        with open(data_yaml, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)
        test_path = data_config.get("test", data_config.get("val"))
        logger.info(f"Evaluating on {split} split: {test_path}")
    else:
        test_path = None

    # Run validation
    logger.info("Running evaluation...")
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        save_dir=str(save_dir) if save_dir else None,
        plots=True,
    )

    # Extract metrics from results
    metrics = {
        "mAP_0.5": float(results.box.map50),
        "mAP_0.5_0.95": float(results.box.map),
        "fitness": float(results.fitness),
    }

    # Per-class metrics
    if hasattr(results.box, "ap_class_index"):
        ap_per_class = results.box.ap
        metrics["ap_per_class"] = {
            str(results.names[int(i)]): float(ap)
            for i, ap in zip(results.box.ap_class_index, ap_per_class)
        }
        # Add mAP@0.5 per class if available
        if hasattr(results.box, "map50_per_class"):
            metrics["map50_per_class"] = {
                str(results.names[int(i)]): float(ap)
                for i, ap in enumerate(results.box.map50_per_class)
                if not np.isnan(ap)
            }

    logger.info(format_metrics_table(metrics))
    return metrics


def analyze_predictions(
    weights_path: Path,
    data_yaml: Path,
    split: str = "test",
    imgsz: int = 1280,
    conf: float = 0.25,
    device: str = "0",
) -> Dict:
    """
    Run inference on the test set and analyze predictions by object size.

    Args:
        weights_path: Path to model weights.
        data_yaml: Path to dataset.yaml.
        split: Dataset split.
        imgsz: Inference image size.
        conf: Confidence threshold for predictions.
        device: GPU device.

    Returns:
        Size breakdown analysis dictionary.
    """
    from ultralytics import YOLO

    # Load data config
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    test_path = data_config.get(split, data_config.get("val"))
    nc = data_config.get("nc", 5)
    names = data_config.get("names", [str(i) for i in range(nc)])

    logger.info(f"Analyzing predictions on {split} split...")
    model = YOLO(str(weights_path))

    # Run inference
    results = model.predict(
        source=test_path,
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=False,
    )

    # Collect prediction stats
    total_predictions = 0
    size_counts = {"small": 0, "medium": 0, "large": 0}
    per_class_counts = {name: 0 for name in names}

    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        total_predictions += len(boxes)
        size_masks = classify_by_area(boxes)

        for size in size_counts:
            size_counts[size] += int(size_masks[size].sum())

        for cls_id in cls_ids:
            if cls_id < len(names):
                per_class_counts[names[cls_id]] += 1

    analysis = {
        "total_predictions": total_predictions,
        "size_distribution": size_counts,
        "per_class_predictions": per_class_counts,
        "conf_threshold": conf,
    }

    logger.info(f"Total predictions: {total_predictions}")
    logger.info(f"Size distribution: {size_counts}")
    logger.info(f"Per-class: {per_class_counts}")

    return analysis


def compare_with_baseline(
    current_metrics: Dict,
    baseline_metrics_path: Optional[Path],
) -> Dict:
    """
    Compare current metrics against a saved baseline.

    Args:
        current_metrics: Current evaluation metrics.
        baseline_metrics_path: Path to baseline metrics JSON.

    Returns:
        Comparison dictionary with delta values.
    """
    if baseline_metrics_path is None or not baseline_metrics_path.exists():
        logger.info("No baseline metrics file provided — skipping comparison.")
        return {}

    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    comparison = {}
    for key in ["mAP_0.5", "mAP_0.5_0.95"]:
        if key in current_metrics and key in baseline:
            delta = current_metrics[key] - baseline[key]
            comparison[f"{key}_delta"] = delta
            logger.info(f"  {key}: {baseline[key]:.4f} → {current_metrics[key]:.4f} "
                        f"({'↑' if delta > 0 else '↓'}{abs(delta):.4f})")

    if "ap_per_class" in current_metrics and "ap_per_class" in baseline:
        comparison["per_class_delta"] = {}
        for cls, ap in current_metrics["ap_per_class"].items():
            if cls in baseline["ap_per_class"]:
                delta = ap - baseline["ap_per_class"][cls]
                comparison["per_class_delta"][cls] = delta

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model for small target detection"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to model weights (.pt file)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/dataset.yaml"),
        help="Path to dataset.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Image size for inference",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device index or 'cpu'",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for validation",
    )
    parser.add_argument(
        "--analyze_sizes",
        action="store_true",
        help="Run detailed size-based analysis of predictions",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline metrics JSON for comparison",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    # Resolve save directory
    if args.save_dir is None:
        args.save_dir = Path("experiments") / "eval_results"
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    metrics = run_evaluation(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        save_dir=args.save_dir,
    )

    # Save metrics
    metrics_path = args.save_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")

    # Compare with baseline if provided
    if args.baseline:
        comparison = compare_with_baseline(metrics, args.baseline)
        if comparison:
            comparison_path = args.save_dir / "comparison.json"
            with open(comparison_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"Comparison saved to {comparison_path}")

    # Size analysis
    if args.analyze_sizes:
        size_analysis = analyze_predictions(
            weights_path=args.weights,
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            conf=0.25,
            device=args.device,
        )
        size_analysis_path = args.save_dir / "size_analysis.json"
        with open(size_analysis_path, "w", encoding="utf-8") as f:
            json.dump(size_analysis, f, indent=2)
        logger.info(f"Size analysis saved to {size_analysis_path}")


if __name__ == "__main__":
    main()
