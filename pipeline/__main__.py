"""Entry point: python -m pipeline <command>."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("yolo-export", help="convert UA-DETRAC XML annotations to a YOLO dataset")
    sub.add_parser("yolo-preview", help="draw labels on sample frames per weather condition")

    train = sub.add_parser("train", help="fine-tune the YOLOv9 checkpoint on data/yolo")
    train.add_argument("--weights", type=Path, default=None, help="starting weights (default: weights/yolov9_vehicle_detection_best.pt)")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16, help="lower it if the GPU runs out of memory")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--oversample", type=int, default=2, help="repeat night and rainy train images this many times")
    train.add_argument("--fraction", type=float, default=1.0, help="use only this fraction of train images (smoke tests)")
    train.add_argument("--name", help="run folder name under data/runs (default: finetune-<timestamp>)")
    train.add_argument("--resume", type=Path, help="path to a last.pt to resume an interrupted run")

    args = parser.parse_args()
    if args.command == "yolo-export":
        from . import yolo_export
        yolo_export.export()
    elif args.command == "yolo-preview":
        from . import yolo_export
        yolo_export.preview()
    else:
        from . import train as trainer
        args.weights = args.weights or trainer.BASE_WEIGHTS
        trainer.train(args)


if __name__ == "__main__":
    main()
