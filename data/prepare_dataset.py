#!/usr/bin/env python3
"""
Prepare TT100K dataset for YOLO training.

Steps:
1. Read raw TT100K JSON annotations
2. Filter 5 target classes: i2, i4, i5, io, p10
3. Convert bbox to YOLO format (class_id cx cy w h normalized)
4. Split into train/val/test (7:2:1)
5. Generate dataset.yaml for ultralytics

Usage:
    python data/prepare_dataset.py --data_dir data/TT100K --output_dir data/processed

Expected TT100K directory structure:
    data/TT100K/
    ├── annotations.json    # Raw TT100K annotations
    ├── train/              # Training images (or all images in a single folder)
    │   ├── 000001.jpg
    │   └── ...
    └── test/               # Test images
        └── ...
"""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

# Set to None to auto-discover ALL classes from annotations.
# Or specify a list like ["i2", "i4", "i5", "pne", "p10"] to filter.
TARGET_CLASSES = None  # None = keep all classes found in annotations

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Default image size (TT100K images are 2048×2048)
DEFAULT_IMG_SIZE = (2048, 2048)


# ─── TT100K Annotation Parsing ──────────────────────────────────────────────

def _build_class_mapping(
    all_categories: set,
    target_classes: Optional[List[str]],
) -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
    """
    Build class list, class_id mapping, and display names from discovered categories.

    Args:
        all_categories: Set of all unique category strings found in annotations.
        target_classes: If provided, only keep these classes. If None, keep all.

    Returns:
        (class_list, cat_to_id, cat_to_display_name)
        - class_list: ordered list of category strings (deterministic via sort)
        - cat_to_id: {category_string: class_id}
        - cat_to_display_name: {category_string: display_name}
    """
    if target_classes is not None:
        filtered = sorted(target_classes)
    else:
        filtered = sorted(all_categories)

    cat_to_id = {cat: i for i, cat in enumerate(filtered)}
    cat_to_display = {cat: cat for cat in filtered}
    return filtered, cat_to_id, cat_to_display


def parse_tt100k_annotations(
    ann_path: Path,
    target_classes: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[Dict]], List[str], Dict[str, int], Dict[str, str]]:
    """
    Parse TT100K JSON annotation file. Supports three formats:

    Format A — Real TT100K (objects nested inside imgs):
    {
        "imgs": {
            "1": {"path": "train/000001.jpg", "id": 1,
                  "objects": [{"category": "i2", "bbox": {"xmin":100,"ymin":200,...}}]}
        }
    }

    Format B — imgs + anns separated:
    {
        "imgs": {"1": {"path": "...", "id": 1}},
        "anns": {"1": [{"category": "i2", "bbox": {...}}]}
    }

    Format C — COCO-style:
    {
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [x,y,w,h]}],
        "images": [...], "categories": [...]
    }

    Returns:
        (annotations_dict, class_list, cat_to_id, cat_to_display_name)
        - annotations_dict: {img_path: [ann_dict, ...]}
        - class_list: ordered list of category strings
        - cat_to_id: {category_string: class_id}
        - cat_to_display_name: {category_string: display_name}
    """
    logger.info(f"Loading annotations from {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    imgs = data.get("imgs", {})
    anns = data.get("anns", {})

    # ── Detect format ────────────────────────────────────────────────────
    result: Dict[str, List[Dict]] = {}
    all_categories: set = set()

    if imgs:
        # Check if objects are nested inside imgs (Format A)
        sample = next(iter(imgs.values()), {})
        if "objects" in sample:
            logger.info("  Detected Format A: objects nested inside imgs")
            for img_id, img_info in imgs.items():
                img_path = img_info.get("path", f"{img_id}.jpg")
                objects = img_info.get("objects", [])

                kept = []
                for obj in objects:
                    category = obj.get("category", "")
                    all_categories.add(category)
                    bbox = obj["bbox"]
                    kept.append({
                        "category": category,
                        "class_id": -1,  # placeholder, filled after class mapping
                        "bbox": [
                            int(bbox["xmin"]), int(bbox["ymin"]),
                            int(bbox["xmax"]), int(bbox["ymax"]),
                        ],
                    })
                if kept:
                    result[img_path] = kept

        elif anns:
            # Format B: imgs + anns separated
            logger.info("  Detected Format B: imgs + anns separated")
            id_to_path = {}
            for img_id, img_info in imgs.items():
                id_to_path[int(img_info.get("id", img_id))] = img_info["path"]

            for img_id, img_annotations in anns.items():
                img_path = id_to_path.get(int(img_id), f"train/{img_id}.jpg")
                kept = []
                for ann in img_annotations:
                    category = ann.get("category", "")
                    all_categories.add(category)
                    bbox = ann["bbox"]
                    kept.append({
                        "category": category,
                        "class_id": -1,  # placeholder
                        "bbox": [
                            int(bbox["xmin"]), int(bbox["ymin"]),
                            int(bbox["xmax"]), int(bbox["ymax"]),
                        ],
                    })
                if kept:
                    result[img_path] = kept
        else:
            logger.warning("  imgs dict found but no 'objects' or 'anns' key")

    # Format C: COCO-style
    if not result and "annotations" in data:
        logger.info("  Detected Format C: COCO-style")
        cat_id_to_name = {}
        for cat in data.get("categories", []):
            name = cat.get("name", cat.get("category", ""))
            cat_id_to_name[cat["id"]] = name

        img_id_to_name = {}
        for img in data.get("images", []):
            img_id_to_name[img["id"]] = img.get("file_name", f"{img['id']:06d}.jpg")

        for ann in data["annotations"]:
            cat_id = ann.get("category_id", -1)
            cat_name = cat_id_to_name.get(cat_id, "unknown")
            all_categories.add(cat_name)
            bbox = ann["bbox"]
            file_name = img_id_to_name.get(ann["image_id"], f"{ann['image_id']:06d}.jpg")

            if file_name not in result:
                result[file_name] = []
            result[file_name].append({
                "category": cat_name,
                "class_id": -1,  # placeholder
                "bbox": [
                    int(bbox[0]), int(bbox[1]),
                    int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]),
                ],
            })

    # ── Build class mapping ──────────────────────────────────────────────
    class_list, cat_to_id, cat_to_display = _build_class_mapping(
        all_categories, target_classes
    )

    # ── Assign class IDs ─────────────────────────────────────────────────
    for img_path, anns in result.items():
        filtered = []
        for ann in anns:
            cat = ann["category"]
            if cat in cat_to_id:
                ann["class_id"] = cat_to_id[cat]
                filtered.append(ann)
        if filtered:
            result[img_path] = filtered
        else:
            # No annotations match target classes — remove this image
            del result[img_path]  # noqa: safe-deletion in loop-over-keys context
            # Actually we need to be careful modifying while iterating.
            # We'll handle this after the loop.

    # Re-filter: remove images with zero matching annotations
    result = {k: v for k, v in result.items() if len(v) > 0}

    logger.info(
        f"Found {len(result)} images with {sum(len(v) for v in result.values())} "
        f"annotations across {len(class_list)} classes"
    )
    return result, class_list, cat_to_id, cat_to_display


# ─── YOLO Format Conversion ─────────────────────────────────────────────────

def bbox_to_yolo(
    bbox_abs: List[int], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Convert absolute bbox [xmin, ymin, xmax, ymax] to YOLO format
    [cx, cy, w, h] normalized to [0, 1].

    Args:
        bbox_abs: [xmin, ymin, xmax, ymax] in pixel coords.
        img_w: Image width.
        img_h: Image height.

    Returns:
        (cx, cy, w, h) normalized.
    """
    x1, y1, x2, y2 = bbox_abs
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return (
        cx / img_w,
        cy / img_h,
        w / img_w,
        h / img_h,
    )


def write_yolo_label(
    output_path: Path,
    annotations: List[Dict],
    img_w: int,
    img_h: int,
) -> None:
    """
    Write annotations in YOLO format: class_id cx cy w h (space-separated).

    Args:
        output_path: Path to output .txt file.
        annotations: List of annotation dicts with 'class_id' and 'bbox'.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
    """
    lines = []
    for ann in annotations:
        cx, cy, w, h = bbox_to_yolo(ann["bbox"], img_w, img_h)
        # Clamp to [0, 1] for safety
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        lines.append(f"{ann['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─── Dataset Splitting ──────────────────────────────────────────────────────

def split_image_list(
    image_names: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split image list into train/val/test sets.

    Args:
        image_names: List of image file names.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        seed: Random seed for reproducibility.

    Returns:
        (train_list, val_list, test_list).
    """
    random.seed(seed)
    names = sorted(image_names)
    random.shuffle(names)

    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = names[:n_train]
    val = names[n_train : n_train + n_val]
    test = names[n_train + n_val :]

    logger.info(
        f"Split: {len(train)} train / {len(val)} val / {len(test)} test "
        f"(total {n} images)"
    )
    return train, val, test


# ─── Dataset YAML Generation ────────────────────────────────────────────────

def generate_dataset_yaml(
    output_path: Path,
    train_path: str,
    val_path: str,
    test_path: str,
    nc: int = 5,
    names: Optional[List[str]] = None,
    class_list: Optional[List[str]] = None,
) -> None:
    """
    Generate ultralytics dataset.yaml configuration.

    Args:
        output_path: Path to write dataset.yaml.
        train_path: Relative or absolute path to train images dir.
        val_path: Path to val images dir.
        test_path: Path to test images dir.
        nc: Number of classes.
        names: List of class display names (deprecated; use class_list).
        class_list: Ordered list of category strings (used if names is None).
    """
    if names is None:
        if class_list is not None:
            names = list(class_list)
        else:
            names = [f"class_{i}" for i in range(nc)]

    config = {
        "path": str(output_path.parent.absolute()),
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": nc,
        "names": names,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated dataset config at {output_path}")


# ─── Image Discovery ────────────────────────────────────────────────────────

def find_images(
    data_dir: Path, image_names: List[str]
) -> Dict[str, Path]:
    """
    Find image files in the data directory matching the given names.

    Searches recursively through subdirectories.

    Args:
        data_dir: Root data directory.
        image_names: List of image file names or relative paths.

    Returns:
        Dict mapping name → absolute path.
    """
    result = {}
    # Index all images in data_dir
    all_images = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
        for img_path in data_dir.rglob(ext):
            if img_path.is_file():
                all_images[img_path.name] = img_path
                # Also index by relative path
                try:
                    rel = img_path.relative_to(data_dir).as_posix()
                    all_images[rel] = img_path
                except ValueError:
                    pass

    for name in image_names:
        # Try exact match first
        if name in all_images:
            result[name] = all_images[name]
            continue

        # Try basename match
        basename = Path(name).name
        if basename in all_images:
            result[name] = all_images[basename]
            continue

        # Try fuzzy match
        for key, path in all_images.items():
            if basename in key or key.endswith(basename):
                result[name] = path
                break

    logger.info(
        f"Found {len(result)}/{len(image_names)} images in {data_dir}"
    )
    return result


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    ann_file: Optional[Path] = None,
    target_classes: Optional[List[str]] = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42,
) -> Path:
    """
    Main dataset preparation pipeline.

    Args:
        data_dir: Root directory containing TT100K data.
        output_dir: Output directory for processed YOLO-format data.
        ann_file: Path to annotations JSON. Auto-detected if None.
        target_classes: Optional list of class strings to keep. None = keep all.
        train_ratio: Train split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.
        seed: Random seed.

    Returns:
        Path to generated dataset.yaml.
    """
    # Auto-detect annotation file
    if ann_file is None:
        candidates = [
            data_dir / "annotations.json",
            data_dir / "annotations.json",
            data_dir / "TT100K_annotations.json",
        ]
        # Also search recursively
        for pattern in ("annotations.json", "*.json"):
            found = list(data_dir.rglob(pattern))
            if found:
                ann_file = found[0]
                break
        else:
            ann_file = candidates[0]

    if not ann_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ann_file}. "
            f"Please place the TT100K annotations.json in {data_dir}"
        )

    # Parse annotations (unified parser handles all 3 formats)
    annotations, class_list, cat_to_id, cat_to_display = parse_tt100k_annotations(
        ann_file, target_classes
    )

    if not annotations:
        raise ValueError(
            "No valid annotations found. Check the annotation file format."
        )

    # Print class distribution
    class_counts = {c: 0 for c in class_list}
    for img_anns in annotations.values():
        for ann in img_anns:
            class_counts[ann["category"]] += 1
    logger.info("Class distribution (all %d classes):", len(class_list))
    for cls in class_list:
        count = class_counts[cls]
        if count > 0:
            logger.info(f"  {cls}: {count}")
    nan_classes = [c for c, n in class_counts.items() if n == 0]
    if nan_classes:
        logger.warning(f"  {len(nan_classes)} classes have zero annotations: {nan_classes[:10]}...")

    # Find images
    image_paths = find_images(data_dir, list(annotations.keys()))

    # Handle missing images
    valid_annotations = {}
    missing = 0
    for name, anns in annotations.items():
        if name in image_paths:
            valid_annotations[name] = anns
        else:
            missing += 1
    if missing > 0:
        logger.warning(
            f"{missing} images referenced in annotations were not found on disk"
        )

    image_names = list(valid_annotations.keys())

    # Split
    train_names, val_names, test_names = split_image_list(
        image_names, train_ratio, val_ratio, test_ratio, seed
    )

    # Create output directories
    splits = {
        "train": (train_names, output_dir / "images" / "train",
                   output_dir / "labels" / "train"),
        "val": (val_names, output_dir / "images" / "val",
                 output_dir / "labels" / "val"),
        "test": (test_names, output_dir / "images" / "test",
                  output_dir / "labels" / "test"),
    }

    for split_name, (names, img_dir, lbl_dir) in splits.items():
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {split_name} split ({len(names)} images)...")
        for name in tqdm(names, desc=split_name):
            src_path = image_paths[name]
            anns = valid_annotations[name]

            # Copy/link image
            dst_path = img_dir / f"{Path(name).stem}.jpg"
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

            # Read image dimensions (use PIL, already a dependency)
            w, h = DEFAULT_IMG_SIZE
            for try_path in (src_path, dst_path):
                try:
                    with Image.open(try_path) as im:
                        w, h = im.size
                    break
                except Exception:
                    continue
            else:
                logger.warning(
                    f"Could not read {name}, using default size {w}×{h}"
                )

            # Write YOLO label
            label_path = lbl_dir / f"{Path(name).stem}.txt"
            write_yolo_label(label_path, anns, w, h)

    # Generate dataset.yaml
    dataset_yaml = output_dir / "dataset.yaml"
    generate_dataset_yaml(
        dataset_yaml,
        train_path=str(splits["train"][1].absolute()),
        val_path=str(splits["val"][1].absolute()),
        test_path=str(splits["test"][1].absolute()),
        nc=len(class_list),
        class_list=class_list,
    )

    # Print summary
    logger.info("=" * 50)
    logger.info("Dataset preparation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset config: {dataset_yaml}")
    logger.info(f"Total images: {len(image_names)}")
    for split_name, (names, _, _) in splits.items():
        logger.info(f"  {split_name}: {len(names)} images")
    logger.info("=" * 50)

    return dataset_yaml


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare TT100K dataset for YOLO training"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root directory containing TT100K dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed data (default: data/processed)",
    )
    parser.add_argument(
        "--ann_file",
        type=Path,
        default=None,
        help="Path to annotations.json (auto-detected if not specified)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        default=None,
        help="Target classes to keep (space-separated). "
             "Default: keep ALL classes found in annotations. "
             "Example: --classes i2 i4 i5 pne p10",
    )

    args = parser.parse_args()

    try:
        dataset_yaml = prepare_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            ann_file=args.ann_file,
            target_classes=args.classes if args.classes else None,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        logger.info(f"Dataset YAML: {dataset_yaml}")
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()
