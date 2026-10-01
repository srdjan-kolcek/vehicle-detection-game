"""Compare two weights on the val split, per weather condition, as a single class (vehicle).

The stock checkpoint has 3 classes and our labels have 1, so single_cls is used: every
prediction counts as a vehicle and NMS is class-agnostic, like detect_sample.py.
"""
import csv
from collections import defaultdict

from ultralytics import YOLO

from .train import RUNS, YOLO_DIR


def write_subsets():
    """Write one image list and dataset yaml per weather (plus all) from the val rows of the manifest."""
    with open(YOLO_DIR / "manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    by_weather = defaultdict(list)
    for row in rows:
        by_weather[row["weather"]].append((YOLO_DIR / row["image"]).as_posix())
    by_weather["all"] = [p for paths in list(by_weather.values()) for p in paths]

    yamls = {}
    for weather, paths in by_weather.items():
        (YOLO_DIR / f"val_{weather}.txt").write_text("\n".join(paths) + "\n")
        yaml_path = YOLO_DIR / f"val_{weather}.yaml"
        yaml_path.write_text(
            f"path: {YOLO_DIR.as_posix()}\ntrain: images/train\nval: val_{weather}.txt\nnames:\n  0: vehicle\n"
        )
        yamls[weather] = (yaml_path, len(paths))
    return yamls


def evaluate(weights, tag, yamls):
    model = YOLO(str(weights))
    results = {}
    for weather, (yaml_path, _) in yamls.items():
        box = model.val(
            data=str(yaml_path),
            imgsz=640,
            batch=2,
            workers=2,
            device=0,
            single_cls=True,
            plots=False,
            verbose=False,
            project=str(RUNS / "val-compare"),
            name=f"{tag}-{weather}",
            exist_ok=True,
        ).box
        results[weather] = (box.mp, box.mr, box.map50, box.map)
        print(f"{tag:5s} {weather:7s} P {box.mp:.3f} R {box.mr:.3f} mAP50 {box.map50:.3f} mAP50-95 {box.map:.3f}", flush=True)
    return results


def compare(args):
    yamls = write_subsets()
    stock = evaluate(args.stock, "stock", yamls)
    new = evaluate(args.new, "new", yamls)

    out = RUNS / "val-compare" / "summary.csv"
    lines = [f"{'weather':8s} {'images':>6s} | {'stock mAP50-95':>14s} {'new mAP50-95':>12s} {'change':>7s} | "
             f"{'stock R':>7s} {'new R':>6s} | {'stock P':>7s} {'new P':>6s}"]
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["weather", "images", "metric", "stock", "new"])
        for weather, (_, count) in yamls.items():
            s, n = stock[weather], new[weather]
            for name, i in (("precision", 0), ("recall", 1), ("mAP50", 2), ("mAP50-95", 3)):
                writer.writerow([weather, count, name, f"{s[i]:.4f}", f"{n[i]:.4f}"])
            lines.append(f"{weather:8s} {count:6d} | {s[3]:14.3f} {n[3]:12.3f} {n[3] - s[3]:+7.3f} | "
                         f"{s[1]:7.3f} {n[1]:6.3f} | {s[0]:7.3f} {n[0]:6.3f}")
    print("\n" + "\n".join(lines))
    print(f"\nsaved: {out}")
