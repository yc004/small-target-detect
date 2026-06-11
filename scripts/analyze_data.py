#!/usr/bin/env python3
"""
Analyze the prepared YOLO dataset for small target statistics.

Computes:
- Per-class object counts and distribution
- Size distribution (small/medium/large based on pixel area)
- Image-level statistics (objects per image, min/max/mean box sizes)
- Generates summary plots

Usage:
    python scripts/analyze_data.py --data_dir data/processed --img_size 1280 1280
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SMALL_THRESH = 32 * 32
MEDIUM_THRESH = 96 * 96


def analyze_labels(
    label_dir: Path, class_names: List[str], img_size: Tuple[int, int]
) -> Dict:
    """
    Analyze YOLO-format labels in a directory.

    Args:
        label_dir: Directory containing .txt label files.
        class_names: List of class name strings.
        img_size: (width, height) of the original images.

    Returns:
        Dict with detailed statistics.
    """
    img_w, img_h = img_size
    total_boxes = 0
    per_class = {name: 0 for name in class_names}
    sizes = {"small": 0, "medium": 0, "large": 0}
    boxes_per_image = []
    box_widths = []
    box_heights = []
    box_areas = []
    class_areas = {name: [] for name in class_names}

    for label_file in sorted(label_dir.glob("*.txt")):
        if not label_file.is_file():
            continue
        n_boxes = 0
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                w_px = w_norm * img_w
                h_px = h_norm * img_h
                area_px = w_px * h_px

                total_boxes += 1
                n_boxes += 1
                if cls_id < len(class_names):
                    per_class[class_names[cls_id]] += 1
                    class_areas[class_names[cls_id]].append(area_px)

                if area_px < SMALL_THRESH:
                    sizes["small"] += 1
                elif area_px < MEDIUM_THRESH:
                    sizes["medium"] += 1
                else:
                    sizes["large"] += 1

                box_widths.append(w_px)
                box_heights.append(h_px)
                box_areas.append(area_px)

        boxes_per_image.append(n_boxes)

    stats = {
        "total_images": len(boxes_per_image),
        "total_boxes": total_boxes,
        "per_class": per_class,
        "size_distribution": sizes,
        "size_distribution_pct": {
            k: (v / total_boxes * 100 if total_boxes > 0 else 0)
            for k, v in sizes.items()
        },
        "boxes_per_image": {
            "min": min(boxes_per_image) if boxes_per_image else 0,
            "max": max(boxes_per_image) if boxes_per_image else 0,
            "mean": np.mean(boxes_per_image) if boxes_per_image else 0,
            "median": np.median(boxes_per_image) if boxes_per_image else 0,
        },
    }

    if box_areas:
        stats["box_pixel_stats"] = {
            "width_mean": float(np.mean(box_widths)),
            "width_median": float(np.median(box_widths)),
            "height_mean": float(np.mean(box_heights)),
            "height_median": float(np.median(box_heights)),
            "area_min": float(np.min(box_areas)),
            "area_max": float(np.max(box_areas)),
            "area_mean": float(np.mean(box_areas)),
            "area_median": float(np.median(box_areas)),
        }

        # Per-class average area
        stats["per_class_avg_area"] = {}
        for name, areas in class_areas.items():
            if areas:
                stats["per_class_avg_area"][name] = float(np.mean(areas))

    return stats


def plot_analysis(stats: Dict, save_dir: Path) -> None:
    """Generate analysis plots and save to save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-class bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    classes = list(stats["per_class"].keys())
    counts = list(stats["per_class"].values())
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    axes[0].bar(classes, counts, color=colors[:len(classes)])
    axes[0].set_title("Objects per Class")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[0].text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=9)

    # 2. Size distribution pie
    sizes = stats["size_distribution"]
    size_labels = list(sizes.keys())
    size_vals = list(sizes.values())
    size_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    axes[1].pie(
        size_vals, labels=size_labels, colors=size_colors[:len(size_labels)],
        autopct="%1.1f%%", startangle=90,
    )
    axes[1].set_title("Object Size Distribution\n(small < 32², medium < 96²)")

    plt.tight_layout()
    fig.savefig(save_dir / "class_size_distribution.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {save_dir / 'class_size_distribution.png'}")

    # 3. Box area histogram
    if "box_pixel_stats" in stats:
        fig, ax = plt.subplots(figsize=(10, 5))
        # We'll read the raw data again from the label files for the histogram
        # For simplicity, show the summary stats as text
        ax.axis("off")
        text = "Box Pixel Statistics:\n\n"
        for k, v in stats["box_pixel_stats"].items():
            text += f"  {k}: {v:.1f}\n"
        text += f"\nSmall target (< {SMALL_THRESH} px²): {stats['size_distribution']['small']} "
        text += f"({stats['size_distribution_pct']['small']:.1f}%)"
        ax.text(0.1, 0.5, text, fontsize=12, fontfamily="monospace",
                verticalalignment="center")
        ax.set_title("Bounding Box Statistics")

        fig.savefig(save_dir / "box_statistics.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {save_dir / 'box_statistics.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset labels for small target statistics"
    )
    parser.add_argument(
        "--label_dir",
        type=Path,
        required=True,
        help="Path to directory with YOLO-format .txt labels",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=["speed_limit_5", "speed_limit_30", "speed_limit_40",
                  "no_entry", "no_pedestrians"],
        help="Class names in order",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[1280, 1280],
        metavar=("WIDTH", "HEIGHT"),
        help="Image dimensions in pixels",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("runs/data_analysis"),
        help="Directory to save analysis outputs",
    )

    args = parser.parse_args()

    if not args.label_dir.exists():
        logger.error(f"Label directory not found: {args.label_dir}")
        return

    logger.info(f"Analyzing labels in {args.label_dir}")
    stats = analyze_labels(args.label_dir, args.class_names, tuple(args.img_size))

    # Print summary
    logger.info("=" * 50)
    logger.info("Dataset Analysis Summary")
    logger.info("=" * 50)
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Total boxes:  {stats['total_boxes']}")
    logger.info(f"  Avg boxes/image: {stats['boxes_per_image']['mean']:.1f}")
    logger.info(f"  Small targets (< 32² px): {stats['size_distribution']['small']} "
                f"({stats['size_distribution_pct']['small']:.1f}%)")
    logger.info(f"  Medium targets (32²-96² px): {stats['size_distribution']['medium']} "
                f"({stats['size_distribution_pct']['medium']:.1f}%)")
    logger.info(f"  Large targets (> 96² px): {stats['size_distribution']['large']} "
                f"({stats['size_distribution_pct']['large']:.1f}%)")

    logger.info("  Per-class counts:")
    for cls, count in stats["per_class"].items():
        avg_area = stats.get("per_class_avg_area", {}).get(cls, 0)
        logger.info(f"    {cls}: {count} (avg area: {avg_area:.0f} px²)")

    if "box_pixel_stats" in stats:
        bs = stats["box_pixel_stats"]
        logger.info(f"  Box size: {bs['width_mean']:.1f}×{bs['height_mean']:.1f} px (mean)")
        logger.info(f"  Box area: min={bs['area_min']:.0f}, max={bs['area_max']:.0f}, "
                    f"median={bs['area_median']:.0f} px²")
    logger.info("=" * 50)

    # Generate plots
    plot_analysis(stats, args.save_dir)

    # Save stats JSON
    import json
    stats_path = args.save_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=float)
    logger.info(f"Statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
