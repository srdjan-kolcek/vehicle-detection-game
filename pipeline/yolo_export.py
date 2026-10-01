"""Convert UA-DETRAC XML annotations to a YOLO dataset with a single class: vehicle.

Every FRAME_STEP-th frame of each Train sequence is exported. Ignored regions are filled
with grey so unlabelled vehicles there are not learned as background. Sequences are split
into train/val by weather, never by frame. The Test sequences are not exported: they stay
held out for the accuracy report.
"""
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = RAW / "DETRAC-Images"
TRAIN_XML = RAW / "DETRAC-Train-Annotations-XML"
OUT = ROOT / "data" / "yolo"

FRAME_STEP = 5
VAL_EVERY = 6
GREY = (114, 114, 114)
PREVIEW_PER_WEATHER = 6


def parse_sequence(xml_path):
    """Return (weather, ignored boxes, {frame number: [boxes]}); boxes are (left, top, w, h) pixels."""
    root = ET.parse(xml_path).getroot()
    # the dataset spells this attribute "sence_weather"
    weather = root.find("sequence_attribute").get("sence_weather")
    ignored = [
        tuple(float(b.get(k)) for k in ("left", "top", "width", "height"))
        for b in root.findall("ignored_region/box")
    ]
    targets = {}
    for frame in root.findall("frame"):
        boxes = []
        for box in frame.findall("target_list/target/box"):
            boxes.append(tuple(float(box.get(k)) for k in ("left", "top", "width", "height")))
        targets[int(frame.get("num"))] = boxes
    return weather, ignored, targets


def to_yolo_line(box, width, height):
    """Clip a pixel box to the image and return a normalised label line, or None if degenerate."""
    left, top, w, h = box
    x1, y1 = max(0.0, left), max(0.0, top)
    x2, y2 = min(float(width), left + w), min(float(height), top + h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    return f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def split_sequences(weathers):
    """Every VAL_EVERY-th sequence within each weather goes to val; the rest to train."""
    by_weather = defaultdict(list)
    for name, weather in sorted(weathers.items()):
        by_weather[weather].append(name)
    split = {}
    for names in by_weather.values():
        for i, name in enumerate(names):
            split[name] = "val" if i % VAL_EVERY == 0 else "train"
    return split


def export():
    sequences = {p.stem: parse_sequence(p) for p in sorted(TRAIN_XML.glob("*.xml"))}
    split = split_sequences({name: seq[0] for name, seq in sequences.items()})

    if OUT.exists():
        shutil.rmtree(OUT)
    for kind in ("images", "labels"):
        for part in ("train", "val"):
            (OUT / kind / part).mkdir(parents=True)

    rows = []
    boxes_per_weather = defaultdict(int)
    for name, (weather, ignored, targets) in sequences.items():
        part = split[name]
        for image_path in sorted((IMAGES / name).glob("img*.jpg")):
            number = int(image_path.stem[3:])
            if (number - 1) % FRAME_STEP:
                continue
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]

            lines = [to_yolo_line(b, width, height) for b in targets.get(number, [])]
            lines = [line for line in lines if line]
            boxes_per_weather[weather] += len(lines)

            for left, top, w, h in ignored:
                x1, y1 = max(0, int(left)), max(0, int(top))
                x2, y2 = min(width, int(left + w) + 1), min(height, int(top + h) + 1)
                image[y1:y2, x1:x2] = GREY

            stem = f"{name}_{image_path.stem}"
            cv2.imwrite(str(OUT / "images" / part / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / "labels" / part / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            rows.append((f"images/{part}/{stem}.jpg", name, weather, part))
        print(f"{name} ({weather}, {part}) done")

    with open(OUT / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "sequence", "weather", "split"])
        writer.writerows(rows)

    (OUT / "dataset.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n"
    )

    counts = defaultdict(int)
    for _, _, weather, part in rows:
        counts[(weather, part)] += 1
    print(f"\nimages: {len(rows)}")
    for (weather, part), n in sorted(counts.items()):
        print(f"  {weather:7s} {part:5s} {n}")
    print("boxes per weather:", dict(sorted(boxes_per_weather.items())))
    print(f"dataset: {OUT / 'dataset.yaml'}")


def preview():
    """Draw the labels on random exported frames, one contact sheet per weather condition."""
    with open(OUT / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    by_weather = defaultdict(list)
    for row in rows:
        by_weather[row["weather"]].append(row)

    out_dir = OUT / "preview"
    out_dir.mkdir(exist_ok=True)
    rng = random.Random(0)
    for weather, items in sorted(by_weather.items()):
        tiles = []
        for row in rng.sample(items, min(PREVIEW_PER_WEATHER, len(items))):
            image = cv2.imread(str(OUT / row["image"]))
            height, width = image.shape[:2]
            label_path = OUT / row["image"].replace("images/", "labels/").replace(".jpg", ".txt")
            for line in label_path.read_text().splitlines():
                _, cx, cy, w, h = map(float, line.split())
                x1, y1 = int((cx - w / 2) * width), int((cy - h / 2) * height)
                x2, y2 = int((cx + w / 2) * width), int((cy + h / 2) * height)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, row["image"].split("/")[-1], (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(cv2.resize(image, (width // 2, height // 2)))
        while len(tiles) % 2:
            tiles.append(tiles[0] * 0)
        grid = cv2.vconcat([cv2.hconcat(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        target = out_dir / f"{weather}.jpg"
        cv2.imwrite(str(target), grid)
        print(f"preview: {target}")
