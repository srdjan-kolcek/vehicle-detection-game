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
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--patience", type=int, default=5, help="stop early after this many epochs without val improvement")
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=2, help="2 fits the 4 GB GTX 1650 Ti (4 spills into system RAM and is about 5x slower per image)")
    train.add_argument("--workers", type=int, default=2, help="data loader processes; each one uses system RAM")
    train.add_argument("--oversample", type=int, default=2, help="repeat night and rainy train images this many times")
    train.add_argument("--fraction", type=float, default=1.0, help="use only this fraction of train images (smoke tests)")
    train.add_argument("--name", help="run folder name under data/runs (default: finetune-<timestamp>)")
    train.add_argument("--resume", type=Path, help="path to a last.pt to resume an interrupted run")

    compare = sub.add_parser("val-compare", help="compare two weights on the val split, per weather")
    compare.add_argument("--new", type=Path, required=True, help="fine-tuned weights, e.g. data/runs/<run>/weights/best.pt")
    compare.add_argument("--stock", type=Path, default=None, help="default: weights/yolov9_vehicle_detection_best.pt")

    count_eval = sub.add_parser("count-eval", help="count vehicles crossing a line on the 40 Test sequences vs XML ground truth")
    count_eval.add_argument("--weights", type=Path, required=True, help="weights to evaluate, e.g. data/runs/<run>/weights/best.pt")
    count_eval.add_argument("--tracker", default="bytetrack.yaml",
                            help="ultralytics tracker yaml, or a file name from pipeline/trackers/ (bytetrack_long.yaml, botsort_reid.yaml)")
    count_eval.add_argument("--conf", type=float, default=0.5,
                            help="detection threshold before tracking; 0.1 was tested (also with bytetrack_second_stage.yaml) and did not count better")
    count_eval.add_argument("--imgsz", type=int, default=640)
    count_eval.add_argument("--tag", help="output folder under data/runs/count-eval (default: <tracker>-conf<conf>-img<imgsz>)")
    count_eval.add_argument("--ref-point", type=float, default=0.75,
                            help="counting point on the box, as a fraction of its height from the top (1 = bottom edge, 0.5 = centre); "
                                 "0.75 counted best, the bottom edge is pulled down by reflections on wet roads")
    count_eval.add_argument("--min-side-frames", type=int, default=0,
                            help="a track only counts if seen this many frames on each side of the line (0 = off)")
    count_eval.add_argument("--limit", type=int, default=0, help="only run the first N Test sequences (0 = all 40; for smoke tests)")

    args = parser.parse_args()
    if args.command == "val-compare":
        from . import train as trainer
        from . import val_compare
        args.stock = args.stock or trainer.BASE_WEIGHTS
        val_compare.compare(args)
    elif args.command == "count-eval":
        from . import count_eval as counter
        counter.evaluate(args)
    elif args.command == "yolo-export":
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
