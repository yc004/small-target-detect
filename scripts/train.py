#!/usr/bin/env python3
"""
Unified training script for all experiment stages.

Usage:
    # Stage 1: Baseline training
    python scripts/train.py --config configs/baseline.yaml

    # Stage 2: P2 model training (future)
    python scripts/train.py --config configs/p2.yaml

    # Override specific parameters
    python scripts/train.py --config configs/baseline.yaml --epochs 50 --batch 4
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register ARF, BCEM custom modules before ultralytics parses any YAML
from models.model_builder import register_custom_modules
register_custom_modules()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def merge_cli_args(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with CLI arguments when provided."""
    cli_overrides = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "model": args.model,
        "data": args.data,
        "project": args.project,
        "name": args.name,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value
    return config


def train(config: dict) -> None:
    """
    Run YOLOv8 training with the given configuration.

    Args:
        config: Training configuration dictionary.
    """
    from ultralytics import YOLO

    # Resolve project path relative to project root
    project = config.get("project", "runs")
    if not Path(project).is_absolute():
        project = str(PROJECT_ROOT / project)
        config["project"] = project

    # Log config (mask sensitive keys if any)
    logger.info("Training configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    # Load model
    model_cfg = config.get("model", "yolov8s.yaml")
    pretrained = config.get("pretrained", True)

    logger.info(f"Loading model: {model_cfg}")
    if model_cfg.endswith(".yaml") or model_cfg.endswith(".yml"):
        model = YOLO(model_cfg)
        if pretrained:
            logger.info("Using pretrained weights (default ultralytics behavior)")
    else:
        # Assume it's a .pt checkpoint
        model = YOLO(model_cfg)
        logger.info(f"Loaded checkpoint: {model_cfg}")

    # Train
    logger.info("Starting training...")
    try:
        results = model.train(
            data=config.get("data"),
            epochs=config.get("epochs", 100),
            imgsz=config.get("imgsz", 1280),
            batch=config.get("batch", 8),
            workers=config.get("workers", 8),
            device=config.get("device", 0),
            optimizer=config.get("optimizer", "auto"),
            lr0=config.get("lr0", 0.01),
            lrf=config.get("lrf", 0.01),
            momentum=config.get("momentum", 0.937),
            weight_decay=config.get("weight_decay", 0.0005),
            warmup_epochs=config.get("warmup_epochs", 3),
            warmup_momentum=config.get("warmup_momentum", 0.8),
            warmup_bias_lr=config.get("warmup_bias_lr", 0.1),
            cos_lr=config.get("cos_lr", True),
            patience=config.get("patience", 20),
            mosaic=config.get("mosaic", 1.0),
            mixup=config.get("mixup", 0.0),
            copy_paste=config.get("copy_paste", 0.0),
            hsv_h=config.get("hsv_h", 0.015),
            hsv_s=config.get("hsv_s", 0.7),
            hsv_v=config.get("hsv_v", 0.4),
            degrees=config.get("degrees", 0.0),
            translate=config.get("translate", 0.1),
            scale=config.get("scale", 0.5),
            shear=config.get("shear", 0.0),
            perspective=config.get("perspective", 0.0),
            flipud=config.get("flipud", 0.0),
            fliplr=config.get("fliplr", 0.5),
            conf=config.get("conf", 0.001),
            iou=config.get("iou", 0.7),
            max_det=config.get("max_det", 300),
            amp=config.get("amp", True),
            close_mosaic=config.get("close_mosaic", 10),
            project=config.get("project", "runs/stage1_baseline"),
            name=config.get("name", "train"),
            exist_ok=config.get("exist_ok", True),
            val=config.get("val", True),
            save=config.get("save", True),
            save_period=config.get("save_period", 10),
            plots=config.get("plots", True),
        )

        logger.info(f"Training complete! Results saved to: {results.save_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for small target traffic sign detection"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to YAML config file (default: configs/baseline.yaml)",
    )

    # CLI overrides (higher priority than config file)
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--imgsz", type=int, help="Override image size")
    parser.add_argument("--device", type=str, help="Override device (e.g., '0', 'cpu')")
    parser.add_argument("--model", type=str, help="Override model config/checkpoint path")
    parser.add_argument("--data", type=str, help="Override dataset.yaml path")
    parser.add_argument("--project", type=str, help="Override project directory")
    parser.add_argument("--name", type=str, help="Override experiment name")

    args = parser.parse_args()

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        logger.info(
            "Run 'python data/prepare_dataset.py --data_dir data/TT100K' first."
        )
        sys.exit(1)

    config = load_config(args.config)
    config = merge_cli_args(config, args)

    train(config)


if __name__ == "__main__":
    main()
