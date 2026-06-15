#!/usr/bin/env python3
"""
One-click model export script for all stages.

Automatically registers custom modules (ARF, BCEM), discovers best.pt,
and exports to ONNX / TensorRT / OpenVINO / CoreML / TFLite.

Usage:
    # Export a single stage to ONNX
    python scripts/export_model.py --stage stage1_baseline

    # Export all completed stages
    python scripts/export_model.py --all

    # Export to multiple formats
    python scripts/export_model.py --stage stage3_arf_head --format onnx trt

    # Custom image size and device
    python scripts/export_model.py --stage stage1_baseline --imgsz 320 --device 0
"""

import argparse
import sys
from pathlib import Path

# ── Patch ultralytics before anything else ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_builder import register_custom_modules

register_custom_modules()

from ultralytics import YOLO


# ── Supported formats ────────────────────────────────────────────────────
SUPPORTED_FORMATS = {
    "onnx":       dict(simplify=True, opset=17),
    "engine":     dict(simplify=True, dynamic=False, half=True),   # TensorRT
    "openvino":   dict(half=False),
    "coreml":     dict(nms=True),
    "tflite":     dict(),
    "torchscript": dict(),
    "saved_model": dict(),
}


def export_model(
    weights: Path,
    formats: list[str],
    imgsz: int = 640,
    device: str = "cpu",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Export a single model to specified formats."""
    print(f"\n{'='*60}")
    print(f"  Exporting: {weights}")
    print(f"  Device:    {device}")
    print(f"  Image size: {imgsz}")
    print(f"  Formats:   {', '.join(formats)}")
    print(f"{'='*60}")

    model = YOLO(str(weights))

    if output_dir is None:
        output_dir = weights.parent

    output_paths: dict[str, Path] = {}

    for fmt in formats:
        fmt_kwargs = SUPPORTED_FORMATS.get(fmt, {}).copy()
        print(f"\n  [{fmt.upper()}] Exporting ...")

        try:
            result = model.export(
                format=fmt,
                imgsz=imgsz,
                device=device,
                **fmt_kwargs,
            )
            # result is either a Path or a tuple (path, metadata)
            if isinstance(result, (tuple, list)):
                out_path = Path(result[0])
            else:
                out_path = Path(str(result))

            # Move to output dir if needed
            if output_dir and out_path.parent != output_dir:
                import shutil
                dest = output_dir / out_path.name
                shutil.move(str(out_path), str(dest))
                out_path = dest

            print(f"  [{fmt.upper()}] ✅ {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
            output_paths[fmt] = out_path
        except Exception as e:
            print(f"  [{fmt.upper()}] ❌ {e}")

    return output_paths


def find_stages() -> list[str]:
    """Find all stage directories with a trained best.pt."""
    stages = []
    exp_dir = PROJECT_ROOT / "experiments"
    if not exp_dir.exists():
        return stages

    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir() or d.name == "eval_results":
            continue
        weights = d / "train" / "weights" / "best.pt"
        if weights.exists():
            stages.append(d.name)
    return stages


def main():
    parser = argparse.ArgumentParser(
        description="One-click YOLO model export (ONNX / TensorRT / TFLite / ...)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Stage name (e.g. stage1_baseline) — auto-discovers best.pt",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Direct path to .pt file (alternative to --stage)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all completed stages",
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["onnx"],
        choices=list(SUPPORTED_FORMATS),
        help="Export format(s). Default: onnx",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Static input size. Default: 640",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export. Default: cpu (use '0' for CUDA GPU)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as weights)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available stages with best.pt",
    )

    args = parser.parse_args()

    # ── --list mode ──
    if args.list:
        stages = find_stages()
        if not stages:
            print("No trained stages found.")
            return
        print("\nAvailable stages:")
        for s in stages:
            w = PROJECT_ROOT / "experiments" / s / "train" / "weights" / "best.pt"
            print(f"  {s:30s}  ({w.stat().st_size / 1e6:.1f} MB)")
        return

    # ── Collect targets ──
    targets: list[Path] = []

    if args.weights:
        targets = [args.weights]
    elif args.stage:
        p = PROJECT_ROOT / "experiments" / args.stage / "train" / "weights" / "best.pt"
        if not p.exists():
            print(f"❌ No best.pt found at: {p}")
            sys.exit(1)
        targets = [p]
    elif args.all:
        for s in find_stages():
            p = PROJECT_ROOT / "experiments" / s / "train" / "weights" / "best.pt"
            targets.append(p)
        if not targets:
            print("No trained stages found.")
            sys.exit(1)
    else:
        parser.print_help()
        print("\n💡 Tip: use --list to see available stages, or --stage <name> to export one.")
        sys.exit(1)

    # ── Export ──
    for t in targets:
        export_model(
            weights=t,
            formats=args.format,
            imgsz=args.imgsz,
            device=args.device,
            output_dir=args.output,
        )

    print(f"\n{'='*60}")
    print(f"  ✅ All exports complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
