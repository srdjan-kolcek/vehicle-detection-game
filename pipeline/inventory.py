"""Camera angles, counting lines and the clip inventory for the demo pool.

Angles: pipeline/angles.yaml maps each city to its angles. One angle is one UA-DETRAC Test
sequence (a fixed camera) and the angle id is the sequence name. Train sequences are refused:
the weights were fine-tuned on them, so their counts could not be verified honestly.

Lines (python -m pipeline lines): one pipeline/lines/<angle>.json per angle, copied from a
count-eval run. Those are the lines the accuracy numbers were measured on, placed automatically
from the predicted tracks. A vehicle counts once when its counting point crosses the segment,
in either direction. Existing line files are kept unless --force is given, so a line edited by
hand is not overwritten. The count-eval preview of each line is copied to data/clips/lines/.

Inventory (python -m pipeline inventory): every angle is cut into consecutive 30 s windows from
its first annotated frame (the remainder is dropped), the same windows count-eval scored. One row
per planned clip goes to data/clips/inventory.csv. Every frame of a window must have an image.
A few Test XMLs skip frames inside their range (MVI_39211 skips 94 frames of empty road); like
count-eval, such a frame counts as having no labelled vehicle, and xml_gap_frames records how
many a clip has.
"""
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import yaml

from .count_eval import CLIP_FRAMES, IMAGES, TEST_XML
from .train import RUNS

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = Path(__file__).resolve().parent
ANGLES = PIPELINE / "angles.yaml"
LINES = PIPELINE / "lines"
CLIPS = ROOT / "data" / "clips"
TRAIN_XML = ROOT / "data" / "raw" / "DETRAC-Train-Annotations-XML"
DEFAULT_LINES_RUN = RUNS / "count-eval" / "bytetrack-conf0.5-img640"

FPS = 25  # UA-DETRAC is recorded at 25 fps; the XML does not store it
DATASET = "ua-detrac"
LICENCE = "UA-DETRAC, non-commercial demo use only (see pipeline/DATASETS.md)"


def load_angles():
    """Return [(city slug, angle id)] in file order; refuse duplicates and non-Test sequences."""
    data = yaml.safe_load(ANGLES.read_text())
    if data.get("dataset") != DATASET:
        raise SystemExit(f"{ANGLES.name}: expected dataset {DATASET}, got {data.get('dataset')}")
    pairs = [(city, angle) for city, angles in data["cities"].items() for angle in angles]
    duplicates = [a for a, n in Counter(a for _, a in pairs).items() if n > 1]
    if duplicates:
        raise SystemExit(f"{ANGLES.name}: angles listed more than once: {', '.join(duplicates)}")
    train = [a for _, a in pairs if (TRAIN_XML / f"{a}.xml").exists()]
    if train:
        raise SystemExit(f"{ANGLES.name}: Train sequences cannot be clips (used for fine-tuning): {', '.join(train)}")
    missing = [a for _, a in pairs if not (TEST_XML / f"{a}.xml").exists()]
    if missing:
        raise SystemExit(f"{ANGLES.name}: no Test XML for: {', '.join(missing)}")
    return pairs


def read_sequence(name):
    """Return (weather, sorted annotated frame numbers) of a Test sequence."""
    root = ET.parse(TEST_XML / f"{name}.xml").getroot()
    # the dataset spells this attribute "sence_weather"
    weather = root.find("sequence_attribute").get("sence_weather")
    return weather, sorted(int(f.get("num")) for f in root.findall("frame"))


def make_lines(args):
    source = Path(args.source)
    LINES.mkdir(exist_ok=True)
    preview_dir = CLIPS / "lines"
    preview_dir.mkdir(parents=True, exist_ok=True)
    written = kept = 0
    for city, angle in load_angles():
        target = LINES / f"{angle}.json"
        if target.exists() and not args.force:
            kept += 1
            continue
        placed = json.loads((source / "lines" / f"{angle}.json").read_text())
        if not placed["p1"]:
            raise SystemExit(f"{angle}: count-eval found no line ({placed['reason']}); draw one by hand in {target}")
        height, width = cv2.imread(str(next((IMAGES / angle).glob("img*.jpg")))).shape[:2]
        line = {
            "angle_id": angle,
            "frame_size": [width, height],
            "p1": [round(v, 1) for v in placed["p1"]],
            "p2": [round(v, 1) for v in placed["p2"]],
            "directions": "both",
            "placed_by": f"auto, count-eval {source.name}",
            "line_share": placed["line_share"],
            "note": placed["reason"],
        }
        target.write_text(json.dumps(line, indent=2) + "\n")
        shutil.copyfile(source / "lines" / f"{angle}.jpg", preview_dir / f"{angle}.jpg")
        written += 1
    print(f"line files written: {written}, kept: {kept} (in {LINES})")
    print(f"line previews: {preview_dir}")


def inventory():
    pairs = load_angles()
    rows = []
    for city, angle in pairs:
        if not (LINES / f"{angle}.json").exists():
            raise SystemExit(f"{angle}: no line file; run python -m pipeline lines first")
        weather, numbers = read_sequence(angle)
        images = {int(p.stem[3:]) for p in (IMAGES / angle).glob("img*.jpg")}
        height, width = cv2.imread(str(IMAGES / angle / f"img{numbers[0]:05d}.jpg")).shape[:2]
        annotated = set(numbers)
        windows = range(numbers[0], numbers[-1] - CLIP_FRAMES + 2, CLIP_FRAMES)
        if not windows:
            raise SystemExit(f"{angle}: shorter than {CLIP_FRAMES} frames, remove it from {ANGLES.name}")
        for first in windows:
            frames = range(first, first + CLIP_FRAMES)
            no_image = [n for n in frames if n not in images]
            if no_image:
                raise SystemExit(f"{angle}: frames {no_image[0]}..{no_image[-1]} have no image")
            rows.append({
                "clip_id": f"{angle}-{first:05d}", "city": city, "angle_id": angle, "sequence": angle,
                "weather": weather, "first_frame": first, "last_frame": first + CLIP_FRAMES - 1,
                "frames": CLIP_FRAMES, "duration_s": CLIP_FRAMES / FPS, "fps": FPS,
                "width": width, "height": height,
                "xml_gap_frames": sum(n not in annotated for n in frames),
                "dataset": DATASET, "licence": LICENCE,
            })

    CLIPS.mkdir(parents=True, exist_ok=True)
    target = CLIPS / "inventory.csv"
    with open(target, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    weathers = sorted({r["weather"] for r in rows})
    print(f"{'city':10s} {'clips':>5s} " + " ".join(f"{w:>7s}" for w in weathers) + "  angles")
    for city in dict.fromkeys(c for c, _ in pairs):
        mine = [r for r in rows if r["city"] == city]
        per = Counter(r["weather"] for r in mine)
        angles = len({r["angle_id"] for r in mine})
        print(f"{city:10s} {len(mine):5d} " + " ".join(f"{per[w]:7d}" for w in weathers) + f"  {angles:6d}")
    per = Counter(r["weather"] for r in rows)
    print(f"{'total':10s} {len(rows):5d} " + " ".join(f"{per[w]:7d}" for w in weathers)
          + f"  {len(pairs):6d}  ({len(rows) * CLIP_FRAMES / FPS / 60:.0f} min)")

    used = {a for _, a in pairs}
    for xml_path in sorted(TEST_XML.glob("*.xml")):
        if xml_path.stem not in used:
            _, numbers = read_sequence(xml_path.stem)
            print(f"not used: {xml_path.stem} ({len(numbers)} frames)")
    print(f"inventory: {target}")
