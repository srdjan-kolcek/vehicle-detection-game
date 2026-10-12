"""Run the YOLOv9 checkpoint on a video or a folder of frames and draw boxes.

Class labels are ignored on purpose: every detection is just a box.
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "weights" / "yolov9_vehicle_detection_best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def read_frames(source):
    if source.is_dir():
        for path in sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            yield cv2.imread(str(path))
        return
    cap = cv2.VideoCapture(str(source))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="video file or folder of frames")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, help="write an annotated mp4 here")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fps", type=float, default=25.0, help="output fps for frame folders")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    writer = None
    frames = 0
    track_ids = set()
    start = time.time()

    for frame in read_frames(args.source):
        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            imgsz=args.imgsz,
            agnostic_nms=True,
            device=0,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes.id is not None:
            track_ids.update(int(i) for i in boxes.id.tolist())

        if args.out:
            for x1, y1, x2, y2 in boxes.xyxy.round().int().tolist():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"boxes: {len(boxes)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if writer is None:
                h, w = frame.shape[:2]
                args.out.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (w, h))
            writer.write(frame)

        frames += 1
        if args.max_frames and frames >= args.max_frames:
            break

    if writer:
        writer.release()
    elapsed = time.time() - start
    print(f"frames: {frames}")
    print(f"unique tracks: {len(track_ids)}")
    print(f"speed: {frames / elapsed:.1f} fps" if elapsed else "speed: n/a")
    if args.out:
        print(f"annotated video: {args.out}")


if __name__ == "__main__":
    main()
