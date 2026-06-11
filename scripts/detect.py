#!/usr/bin/env python3
"""
Inference/detection script for small target traffic sign detection.

Supports: single image, image directory, video file, webcam.

Usage:
    # Single image
    python scripts/detect.py --weights best.pt --source test.jpg

    # Image directory
    python scripts/detect.py --weights best.pt --source data/test_images/

    # Video file
    python scripts/detect.py --weights best.pt --source road_video.mp4

    # Webcam (device 0)
    python scripts/detect.py --weights best.pt --source 0 --conf 0.4
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def detect_images(
    model,
    source: str,
    imgsz: int,
    conf: float,
    iou: float,
    save_dir: Path,
    show: bool = False,
) -> None:
    """Run detection on images."""
    from ultralytics import YOLO as YOLOModel

    logger.info(f"Running inference on: {source}")
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=True,
        save_txt=False,
        save_conf=False,
        project=str(save_dir),
        name="detect",
        exist_ok=True,
    )

    for i, result in enumerate(results):
        if result.boxes is not None:
            n_dets = len(result.boxes)
            logger.info(f"  Image {i+1}: {n_dets} detections")

            # Log per-class counts
            if n_dets > 0:
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                for cls_id in set(cls_ids):
                    name = result.names.get(cls_id, str(cls_id))
                    count = (cls_ids == cls_id).sum()
                    logger.info(f"    {name}: {count}")

    logger.info(f"Results saved to {save_dir / 'detect'}")


def detect_video(
    model,
    source: str,
    imgsz: int,
    conf: float,
    iou: float,
    save_dir: Path,
    show: bool = False,
) -> None:
    """Run detection on video."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {source}")
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    output_path = save_dir / "output_video.mp4"
    save_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    logger.info(
        f"Processing video: {width}×{height} @ {fps:.1f} fps, {total_frames} frames"
    )

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
            annotated = results[0].plot()

            out.write(annotated)

            if show:
                cv2.imshow("Traffic Sign Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                logger.info(
                    f"  Frame {frame_idx}/{total_frames} "
                    f"({100 * frame_idx / total_frames:.1f}%)"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()

    logger.info(f"Output saved to {output_path}")


def detect_webcam(
    model,
    source: int,
    imgsz: int,
    conf: float,
    iou: float,
) -> None:
    """Run real-time detection on webcam feed."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam device {source}")
        return

    logger.info("Starting webcam detection. Press 'q' to quit, 's' to screenshot.")

    fps_counter = []
    try:
        import time

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
            annotated = results[0].plot()

            # FPS display
            fps_counter.append(1.0 / (time.time() - t0 + 1e-6))
            if len(fps_counter) > 50:
                fps_counter.pop(0)
            avg_fps = sum(fps_counter) / len(fps_counter)

            cv2.putText(
                annotated,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Traffic Sign Detection (q=quit, s=save)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(save_path, annotated)
                logger.info(f"Screenshot saved: {save_path}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.info(f"Average FPS: {sum(fps_counter) / len(fps_counter):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run detection with trained YOLO model"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: image path, directory, video file, or webcam index",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (must match training)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device index or 'cpu'",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("runs/detect_results"),
        help="Directory to save detection results",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display detection results in a window",
    )

    args = parser.parse_args()

    from ultralytics import YOLO

    logger.info(f"Loading model from {args.weights}")
    model = YOLO(str(args.weights))

    # Determine source type
    source = args.source
    is_webcam = source.isdigit()
    is_video = source.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    is_image = source.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))

    source_path = Path(source)
    is_dir = source_path.is_dir()

    if is_webcam:
        detect_webcam(
            model,
            source=int(source),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
    elif is_video:
        detect_video(
            model,
            source=source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            save_dir=args.save_dir,
            show=args.show,
        )
    elif is_image or is_dir:
        detect_images(
            model,
            source=source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            save_dir=args.save_dir,
            show=args.show,
        )
    else:
        logger.error(f"Unsupported source: {source}")
        logger.info(
            "Supported: image files, directories, video files (.mp4, .avi, etc.), "
            "webcam index (0, 1, ...)"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
