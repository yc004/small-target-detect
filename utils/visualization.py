"""
Visualization utilities for small target detection analysis.

Features:
- Side-by-side baseline vs improved model comparison
- Size-based color coding of detections
- Feature map visualization
- PR curve plotting
- Confusion matrix visualization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for scripts
logger = logging.getLogger(__name__)

# ─── Color Maps ─────────────────────────────────────────────────────────────

# Distinct colors for up to 10 classes (BGR for OpenCV)
CLASS_COLORS_BGR = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 255),   # Orange
    (255, 128, 0),   # Sky Blue
    (128, 255, 0),   # Lime
    (0, 128, 255),   # Salmon
]

# Size-based colors
SIZE_COLORS_BGR = {
    "small": (0, 0, 255),     # Red — small targets (key focus)
    "medium": (0, 255, 255),   # Yellow
    "large": (0, 255, 0),      # Green
}


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    cls_ids: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    color_by: str = "class",
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Args:
        image: Input image (H, W, 3) in BGR.
        boxes: (N, 4) array in [x1, y1, x2, y2] format.
        cls_ids: (N,) array of class IDs.
        scores: (N,) array of confidence scores.
        class_names: List of class name strings.
        color_by: 'class' or 'size' — determines color scheme.
        line_thickness: Box line thickness.
        font_scale: Text size scale.

    Returns:
        Annotated image (copy).
    """
    img = image.copy()
    h, w = img.shape[:2]

    if boxes is None or len(boxes) == 0:
        return img

    n = len(boxes)

    if color_by == "size":
        from utils.metrics import classify_by_area

        size_masks = classify_by_area(boxes)
        colors = []
        for i in range(n):
            if size_masks["small"][i]:
                colors.append(SIZE_COLORS_BGR["small"])
            elif size_masks["medium"][i]:
                colors.append(SIZE_COLORS_BGR["medium"])
            else:
                colors.append(SIZE_COLORS_BGR["large"])
    else:
        if cls_ids is not None:
            colors = [CLASS_COLORS_BGR[int(c) % len(CLASS_COLORS_BGR)] for c in cls_ids]
        else:
            colors = [CLASS_COLORS_BGR[0]] * n

    for i in range(n):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        color = colors[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        # Label
        label_parts = []
        if cls_ids is not None and class_names is not None:
            cid = int(cls_ids[i])
            if cid < len(class_names):
                label_parts.append(class_names[cid])
        if scores is not None:
            label_parts.append(f"{scores[i]:.2f}")

        if label_parts:
            label = " ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(
                img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1
            )
            cv2.putText(
                img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
            )

    return img


def compare_detections(
    image_path: Path,
    baseline_boxes: np.ndarray,
    improved_boxes: np.ndarray,
    baseline_scores: Optional[np.ndarray] = None,
    improved_scores: Optional[np.ndarray] = None,
    baseline_cls: Optional[np.ndarray] = None,
    improved_cls: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Create a side-by-side comparison of baseline vs improved model detections.

    Args:
        image_path: Path to the source image.
        baseline_boxes: Detection boxes from baseline model.
        improved_boxes: Detection boxes from improved model.
        baseline_scores / improved_scores: Confidence scores.
        baseline_cls / improved_cls: Class IDs.
        class_names: Class name list.
        save_path: Path to save the comparison image.

    Returns:
        Comparison image (numpy array).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_baseline = draw_boxes(img, baseline_boxes, baseline_cls, baseline_scores, class_names)
    img_improved = draw_boxes(img, improved_boxes, improved_cls, improved_scores, class_names)

    # Add titles
    h, w = img.shape[:2]
    title_bar = np.ones((40, w * 2, 3), dtype=np.uint8) * 50
    cv2.putText(title_bar, "Baseline", (w // 2 - 40, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(title_bar, "Improved", (w + w // 2 - 40, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    comparison = np.vstack([title_bar, np.hstack([img_baseline, img_improved])])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), comparison)
        logger.info(f"Comparison saved to {save_path}")

    return comparison


def plot_pr_curves(
    metrics_per_class: Dict[str, Dict[str, List[float]]],
    save_path: Path,
    title: str = "Precision-Recall Curves",
) -> None:
    """
    Plot PR curves for multiple classes.

    Args:
        metrics_per_class: Dict mapping class_name → {'precision': [...], 'recall': [...], 'ap': float}
        save_path: Path to save the plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (cls_name, data) in enumerate(metrics_per_class.items()):
        precision = data.get("precision", [])
        recall = data.get("recall", [])
        ap = data.get("ap", 0.0)
        color = [c / 255.0 for c in CLASS_COLORS_BGR[i % len(CLASS_COLORS_BGR)]][::-1]  # BGR → RGB

        if len(precision) > 0 and len(recall) > 0:
            ax.plot(recall, precision, color=color, linewidth=2,
                    label=f"{cls_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curves saved to {save_path}")


def plot_metrics_comparison(
    baseline_metrics: Dict,
    improved_metrics: Dict,
    metric_names: List[str],
    save_path: Path,
    title: str = "Metrics Comparison",
) -> None:
    """
    Bar chart comparing key metrics between baseline and improved model.

    Args:
        baseline_metrics: Baseline metric values.
        improved_metrics: Improved model metric values.
        metric_names: List of metric keys to compare.
        save_path: Path to save the plot.
        title: Plot title.
    """
    n = len(metric_names)
    x = np.arange(n)
    width = 0.35

    baseline_vals = [baseline_metrics.get(m, 0) for m in metric_names]
    improved_vals = [improved_metrics.get(m, 0) for m in metric_names]
    deltas = [i - b for i, b in zip(improved_vals, baseline_vals)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#3498db")
    bars2 = ax.bar(x + width / 2, improved_vals, width, label="Improved", color="#2ecc71")

    # Add delta annotations
    for i, (b, i_val, d) in enumerate(zip(baseline_vals, improved_vals, deltas)):
        sign = "+" if d > 0 else ""
        ax.annotate(
            f"{sign}{d:.3f}",
            (x[i] + width / 2, max(b, i_val) + 0.01),
            ha="center",
            fontsize=9,
            color="red" if d < 0 else "green",
        )

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Metrics comparison saved to {save_path}")


def plot_size_distribution(
    size_counts: Dict[str, int],
    save_path: Path,
    title: str = "Detection Size Distribution",
) -> None:
    """
    Pie chart showing the distribution of detections by object size.

    Args:
        size_counts: Dict with 'small', 'medium', 'large' → count.
        save_path: Path to save the plot.
        title: Plot title.
    """
    labels = list(size_counts.keys())
    values = list(size_counts.values())
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors[:len(labels)],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax.set_title(title, fontsize=14)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Size distribution saved to {save_path}")
