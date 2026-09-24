"""Fine-tune the YOLOv9 checkpoint on the exported UA-DETRAC dataset (data/yolo/).

Night and rainy training images are repeated so they are seen more often; val is untouched.
Ultralytics keeps best.pt and last.pt in the run folder. The starting weights in weights/ are
only read, never written. One line per epoch is printed and appended to epochs.log.
"""
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "data" / "yolo"
RUNS = ROOT / "data" / "runs"
BASE_WEIGHTS = ROOT / "weights" / "yolov9_vehicle_detection_best.pt"
OVERSAMPLED = {"night", "rainy"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(oversample):
    """Write the train image list (night/rainy repeated) and the dataset yaml; return image counts."""
    with open(YOLO_DIR / "manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    lines = []
    for row in rows:
        repeat = oversample if row["weather"] in OVERSAMPLED else 1
        lines += [(YOLO_DIR / row["image"]).as_posix()] * repeat
    (YOLO_DIR / "train_list.txt").write_text("\n".join(lines) + "\n")
    (YOLO_DIR / "finetune.yaml").write_text(
        f"path: {YOLO_DIR.as_posix()}\ntrain: train_list.txt\nval: images/val\nnames:\n  0: vehicle\n"
    )
    return {"unique_train_images": len(rows), "train_list_entries": len(lines)}


def add_epoch_logger(model):
    """Print and append one line to epochs.log at the end of every epoch."""
    state = {"start": time.time(), "epoch_start": time.time()}

    def on_epoch_start(trainer):
        state["epoch_start"] = time.time()

    def on_epoch_end(trainer):
        metrics = trainer.metrics
        losses = trainer.label_loss_items(trainer.tloss)
        lr = next(iter(trainer.lr.values()), 0.0)
        now = time.time()
        new_best = " *best*" if trainer.fitness == trainer.best_fitness else ""
        line = (
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | epoch {trainer.epoch + 1}/{trainer.epochs}"
            f" | box {losses.get('train/box_loss', 0):.4f} cls {losses.get('train/cls_loss', 0):.4f}"
            f" dfl {losses.get('train/dfl_loss', 0):.4f}"
            f" | val P {metrics.get('metrics/precision(B)', 0):.4f}"
            f" R {metrics.get('metrics/recall(B)', 0):.4f}"
            f" mAP50 {metrics.get('metrics/mAP50(B)', 0):.4f}"
            f" mAP50-95 {metrics.get('metrics/mAP50-95(B)', 0):.4f}"
            f" | lr {lr:.6f} | epoch {now - state['epoch_start']:.0f}s"
            f" total {(now - state['start']) / 60:.1f}min{new_best}"
        )
        print(line, flush=True)
        with open(Path(trainer.save_dir) / "epochs.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_fit_epoch_end", on_epoch_end)


def train(args):
    if args.resume:
        model = YOLO(str(args.resume))
        add_epoch_logger(model)
        model.train(resume=True)
        return

    counts = write_dataset(args.oversample)
    name = args.name or f"finetune-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=False)
    info = {
        "base_weights": str(args.weights),
        "base_weights_sha256": sha256(args.weights),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": 0,
        "oversample_weather": sorted(OVERSAMPLED),
        "oversample": args.oversample,
        "fraction": args.fraction,
        "manifest_sha256": sha256(YOLO_DIR / "manifest.csv"),
        **counts,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)

    model = YOLO(str(args.weights))
    add_epoch_logger(model)
    model.train(
        data=str(YOLO_DIR / "finetune.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        fraction=args.fraction,
        device=0,
        project=str(RUNS),
        name=name,
        exist_ok=True,
        seed=0,
        deterministic=True,
        hsv_v=0.5,
    )

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        print(f"best.pt sha256: {sha256(best)}", flush=True)
        (run_dir / "best_weights_sha256.txt").write_text(sha256(best) + "\n")
