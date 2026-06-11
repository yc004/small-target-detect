#!/usr/bin/env python3
"""
Extract and filter TT100K YOLO-format dataset from ZIP.

Only extracts images that contain the target classes, filters labels
to keep only target classes, and remaps class IDs to 0-based contiguous.

Usage:
    python data/extract_dataset.py
    python data/extract_dataset.py --zip_path /path/to/TT100K-YOLO格式.zip
    python data/extract_dataset.py --zip_path /path/to/data.zip --output_dir /path/to/output

The script reads from Datasets-TT100K/TT100K-YOLO格式.zip and outputs to data/processed/.

Target classes (from the 45-class YOLO subset):
    Class    Original ID    Description
    i2       31             限速5 (speed limit 5)
    i4       12             限速30 (speed limit 30)
    i5       42             限速40 (speed limit 40)
    p10      22             禁止行人通行 (no pedestrians)
    pne      14             禁止驶入 (no entry) — closest to 'io'

Note: 'io' is not in the 45-class TT100K subset. Using 'pne' as the
semantically closest replacement (both mean "no entry").
"""

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "Datasets-TT100K" / "TT100K-YOLO格式.zip"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
YAML_PATH = PROJECT_ROOT / "Datasets-TT100K" / "data_tt100k.yaml"

# Target classes: (original_class_id, new_class_id, class_key, display_name)
# The class_key is the TT100K short name, display_name is human-readable
TARGET_CLASSES: List[Tuple[int, int, str, str]] = [
    (31, 0, "i2", "speed_limit_5"),
    (12, 1, "i4", "speed_limit_30"),
    (42, 2, "i5", "speed_limit_40"),
    (14, 3, "pne", "no_entry"),        # pne = 禁止驶入, replaces 'io'
    (22, 4, "p10", "no_pedestrians"),
]

# Original class names (from data_tt100k.yaml, all 45 classes)
ORIGINAL_CLASS_NAMES = {
    0: "pl80", 1: "p6", 2: "p5", 3: "pm55", 4: "pl60",
    5: "ip", 6: "p11", 7: "i2r", 8: "p23", 9: "pg",
    10: "il80", 11: "ph4", 12: "i4", 13: "pl70", 14: "pne",
    15: "ph4.5", 16: "p12", 17: "p3", 18: "pl5", 19: "w13",
    20: "i4l", 21: "pl30", 22: "p10", 23: "pn", 24: "w55",
    25: "p26", 26: "p13", 27: "pr40", 28: "pl20", 29: "pm30",
    30: "pl40", 31: "i2", 32: "pl120", 33: "w32", 34: "ph5",
    35: "il60", 36: "w57", 37: "pl100", 38: "w59", 39: "il100",
    40: "p19", 41: "pm20", 42: "i5", 43: "p27", 44: "pl50",
}

# Splits to process
SPLITS = ["train", "val", "test"]


# ─── Core Logic ──────────────────────────────────────────────────────────────

def build_target_map() -> Dict[int, int]:
    """Build mapping: original_class_id → new_class_id for target classes only."""
    return {orig_id: new_id for orig_id, new_id, _, _ in TARGET_CLASSES}


def filter_label_content(
    content: str, target_map: Dict[int, int]
) -> Tuple[str, int]:
    """
    Filter a YOLO label file, keeping only target class lines.

    Args:
        content: Raw label file content (one box per line).
        target_map: {original_class_id: new_class_id}.

    Returns:
        (filtered_content, num_boxes_kept). Filtered content is empty string
        if no target boxes found.
    """
    kept_lines = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        if cls_id in target_map:
            new_id = target_map[cls_id]
            parts[0] = str(new_id)
            kept_lines.append(" ".join(parts))

    return "\n".join(kept_lines), len(kept_lines)


def process_split(
    zip_file: zipfile.ZipFile,
    split: str,
    target_map: Dict[int, int],
    output_images_dir: Path,
    output_labels_dir: Path,
) -> Dict:
    """
    Process one data split: filter labels and extract matching images.

    Strategy (disk-efficient):
    1. Iterate over label files in the ZIP for this split
    2. Read each label, filter for target classes
    3. If target boxes found → write new label + extract image
    4. Otherwise → skip (save disk space)

    Returns:
        Statistics dict for this split.
    """
    label_prefix = f"labels/{split}/"
    image_prefix = f"images/{split}/"

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get list of label files in the ZIP for this split
    all_names = zip_file.namelist()
    label_files = sorted(
        n for n in all_names
        if n.startswith(label_prefix) and n.endswith(".txt")
    )

    stats = {
        "total_labels": len(label_files),
        "kept_images": 0,
        "skipped_images": 0,
        "total_boxes_kept": 0,
        "per_class_boxes": {name: 0 for _, _, _, name in TARGET_CLASSES},
        "missing_images": 0,
    }

    logger.info(f"Processing '{split}' split: {len(label_files)} label files...")

    for i, label_path in enumerate(label_files):
        filename = Path(label_path).stem
        image_name = f"{filename}.jpg"
        image_path = f"{image_prefix}{image_name}"

        # Read and filter label
        raw_content = zip_file.read(label_path).decode("utf-8", errors="replace")
        filtered, n_kept = filter_label_content(raw_content, target_map)

        if n_kept == 0:
            stats["skipped_images"] += 1
            continue

        # Check image exists in ZIP
        if image_path not in all_names:
            stats["missing_images"] += 1
            if stats["missing_images"] <= 5:
                logger.warning(f"  Image not found in ZIP: {image_path}")
            continue

        # Count per-class
        for line in filtered.split("\n"):
            if not line.strip():
                continue
            cls_id = int(line.split()[0])
            if cls_id < len(TARGET_CLASSES):
                name = TARGET_CLASSES[cls_id][3]
                stats["per_class_boxes"][name] += 1

        # Write filtered label
        label_out = output_labels_dir / f"{filename}.txt"
        with open(label_out, "w", encoding="utf-8") as f:
            f.write(filtered + "\n")

        # Extract image
        img_out = output_images_dir / image_name
        if not img_out.exists():
            with zip_file.open(image_path) as src:
                with open(img_out, "wb") as dst:
                    dst.write(src.read())

        stats["kept_images"] += 1
        stats["total_boxes_kept"] += n_kept

        # Progress
        if (i + 1) % 500 == 0 or i == len(label_files) - 1:
            logger.info(
                f"  [{i + 1}/{len(label_files)}] "
                f"kept={stats['kept_images']}, "
                f"skipped={stats['skipped_images']}, "
                f"boxes={stats['total_boxes_kept']}"
            )

    return stats


def generate_dataset_yaml(output_dir: Path) -> Path:
    """Generate ultralytics dataset.yaml for the filtered dataset."""
    import yaml

    names = [name for _, _, _, name in TARGET_CLASSES]
    config = {
        "path": str(output_dir.absolute()),
        "train": str((output_dir / "images" / "train").absolute()),
        "val": str((output_dir / "images" / "val").absolute()),
        "test": str((output_dir / "images" / "test").absolute()),
        "nc": len(names),
        "names": names,
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Dataset YAML written to {yaml_path}")
    return yaml_path


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_output(output_dir: Path) -> bool:
    """
    Validate the filtered dataset:
    - Every label file has a corresponding image
    - Every label has valid class IDs (0-4)
    - No empty label files
    """
    valid = True
    for split in SPLITS:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            logger.warning(f"  {split}: directory missing")
            valid = False
            continue

        images = {p.stem for p in img_dir.glob("*.jpg")}
        labels = {p.stem for p in lbl_dir.glob("*.txt")}

        # Every label must have an image
        orphan_labels = labels - images
        if orphan_labels:
            logger.warning(f"  {split}: {len(orphan_labels)} labels without images")
            valid = False

        # Every image should have a label (by construction)
        orphan_images = images - labels
        if orphan_images:
            logger.warning(f"  {split}: {len(orphan_images)} images without labels")
            # Not necessarily invalid, but worth noting

        # Check class IDs
        for lbl_path in lbl_dir.glob("*.txt"):
            content = lbl_path.read_text().strip()
            if not content:
                logger.warning(f"  {split}: empty label {lbl_path.name}")
                valid = False
                continue
            for line in content.split("\n"):
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    if cls_id < 0 or cls_id >= len(TARGET_CLASSES):
                        logger.warning(
                            f"  {split}/{lbl_path.name}: invalid class ID {cls_id}"
                        )
                        valid = False

        logger.info(f"  {split}: {len(images)} images, {len(labels)} labels")

    return valid


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract and filter TT100K YOLO-format dataset from ZIP"
    )
    parser.add_argument(
        "--zip_path", type=Path, default=ZIP_PATH,
        help=f"Path to YOLO-format ZIP (default: {ZIP_PATH})",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory for filtered dataset (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    zip_path = args.zip_path
    output_dir = args.output_dir

    if not zip_path.exists():
        logger.error(f"ZIP file not found: {zip_path}")
        logger.error(
            "Specify the path with --zip_path, e.g.:\n"
            "  python data/extract_dataset.py --zip_path /public/data/image/TT100K/TT100K-YOLO格式.zip"
        )
        sys.exit(1)

    target_map = build_target_map()

    # Print what we're filtering for
    logger.info("=" * 60)
    logger.info("Target classes (original ID → new ID):")
    for orig_id, new_id, key, name in TARGET_CLASSES:
        orig_name = ORIGINAL_CLASS_NAMES.get(orig_id, "?")
        logger.info(f"  {key} ({name}): {orig_id} → {new_id}")
    logger.info("=" * 60)

    # Clean output directory
    if output_dir.exists():
        logger.info(f"Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for split in SPLITS:
            img_dir = output_dir / "images" / split
            lbl_dir = output_dir / "labels" / split

            stats = process_split(zf, split, target_map, img_dir, lbl_dir)
            all_stats[split] = stats

    # ─── Summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Extraction Summary")
    logger.info("=" * 60)

    total_images = 0
    total_boxes = 0
    total_per_class = {name: 0 for _, _, _, name in TARGET_CLASSES}

    for split in SPLITS:
        s = all_stats[split]
        total_images += s["kept_images"]
        total_boxes += s["total_boxes_kept"]
        for name in total_per_class:
            total_per_class[name] += s["per_class_boxes"].get(name, 0)

        logger.info(
            f"  {split:6s}: {s['kept_images']:5d} images, "
            f"{s['total_boxes_kept']:5d} boxes "
            f"(skipped {s['skipped_images']} images without targets)"
        )

    logger.info(f"  {'TOTAL':6s}: {total_images:5d} images, {total_boxes:5d} boxes")
    logger.info("  Per-class box counts:")
    for name, count in total_per_class.items():
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0
        logger.info(f"    {name}: {count} ({pct:.1f}%)")
    logger.info("=" * 60)

    # ─── Generate YAML ───────────────────────────────────────────────────
    generate_dataset_yaml(output_dir)

    # ─── Validate ─────────────────────────────────────────────────────────
    logger.info("\nValidating output...")
    if validate_output(output_dir):
        logger.info("✓ Validation passed!")
    else:
        logger.warning("⚠ Validation found issues — review above warnings")

    logger.info(f"\nDataset ready at: {output_dir}")
    logger.info(f"Config file: {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
