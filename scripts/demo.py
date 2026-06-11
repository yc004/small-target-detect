#!/usr/bin/env python3
"""
Real-time traffic sign detection demo with webcam.

Usage:
    python scripts/demo.py --model best.pt
    python scripts/demo.py --model best.pt --conf 0.5 --imgsz 640
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Webcam demo for traffic sign detection")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.cam}")
        return

    print("Press 'q' to quit, 's' to screenshot")
    fps_list = []

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        # FPS
        dt = time.time() - t0
        fps_list.append(1.0 / dt if dt > 0 else 0)
        if len(fps_list) > 50:
            fps_list.pop(0)
        fps = sum(fps_list) / len(fps_list)

        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Traffic Sign Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            name = time.strftime("screenshot_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(name, annotated)
            print(f"Saved: {name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
