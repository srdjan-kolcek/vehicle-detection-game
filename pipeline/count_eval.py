"""Count vehicles crossing a line on the 40 held-out UA-DETRAC Test sequences and compare
against XML ground truth, per weather condition.

Tracking: the given weights run with ByteTrack over every frame (never subsampled: tracking
needs continuous motion). Ignored regions are greyed out first, same as yolo_export.py, since
the annotators left vehicles there unlabelled. The tracked boxes are cached per sequence in
data/runs/count-eval/tracks/, so changing the line or the counting rule re-scores in seconds.

Line: placed automatically from the predicted tracks, never from the ground truth, so the same
method works on game footage without annotations.
  1. Keep tracks seen for at least MIN_TRACK_FRAMES frames that moved at least MIN_TRAVEL px.
  2. Group them by direction of travel; opposite lanes of one road share an axis. The axis with
     the most tracks is the main flow.
  3. The line is a segment across that axis, at the position most main-flow tracks pass through,
     preferring positions near the camera (bigger boxes) and away from ignored regions. It only
     spans the width of the main flow, so vehicles on other roads do not cross it.
  4. If the main flow has less than MIN_FLOW_SHARE of the tracks (e.g. an intersection), the
     sequence is flagged unsuitable. It is still scored, but reported separately.

Counting: the reference point of a box is its bottom centre. A track counts once, the first time
it moves from one side of the segment to the other within the segment's length, and only if it
was seen for at least --min-side-frames frames on each side. That drops track fragments that start
right at the line. The ground truth is counted on the same line with the same rule.

Box score: tracked boxes are matched to GT boxes frame by frame with the Hungarian method on IoU,
at IoU 0.5 and 0.7. Reported: matched, extra (no GT) and missed (no prediction) boxes, and ID
switches, i.e. a GT vehicle whose matched track id changes, which a line counter sees as a new
vehicle.
"""
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from .train import RUNS

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGES = RAW / "DETRAC-Images"
TEST_XML = RAW / "DETRAC-Test-Annotations-XML"
OUT = RUNS / "count-eval"
TRACKS = OUT / "tracks"
LINES = OUT / "lines"
GREY = (114, 114, 114)

MIN_TRACK_FRAMES = 25  # 1 s at 25 fps
MIN_TRAVEL = 40.0  # px; slower tracks are parked or false
AXIS_BIN_DEG = 10
# a track belongs to the main flow if its axis is this close to it; wide enough for perspective to
# fan out the lanes of one road (about 30 deg apart), narrow enough to split crossing roads (about 90)
AXIS_TOLERANCE_DEG = 35
EDGE_MARGIN = 10  # px; a line this close to the frame edge meets vehicles that are cut off or leaving
MIN_FLOW_SHARE = 0.7
MIN_FLOW_TRACKS = 5
IOU_LEVELS = (0.5, 0.7)


def parse_test_sequence(xml_path):
    """Return (weather, ignored boxes, {frame number: [(target id, x1, y1, x2, y2)]})."""
    root = ET.parse(xml_path).getroot()
    weather = root.find("sequence_attribute").get("sence_weather")
    ignored = [
        tuple(float(b.get(k)) for k in ("left", "top", "width", "height"))
        for b in root.findall("ignored_region/box")
    ]
    targets = {}
    for frame in root.findall("frame"):
        boxes = []
        for target in frame.findall("target_list/target"):
            left, top, w, h = (float(target.find("box").get(k)) for k in ("left", "top", "width", "height"))
            boxes.append((int(target.get("id")), left, top, left + w, top + h))
        targets[int(frame.get("num"))] = boxes
    return weather, ignored, targets


def run_tracker(model, image_dir, ignored, conf, imgsz):
    """Run ByteTrack over every frame in image_dir; return {frame number: [(track id, x1, y1, x2, y2)]}."""
    # ultralytics registers its tracking callbacks once, on the first track() call, and freezes the
    # persist value it got then; persist=False there would rebuild the tracker on every frame. So
    # always pass persist=True and reset the existing tracker between sequences instead.
    for tracker in getattr(model.predictor, "trackers", []):
        tracker.reset()
    frames = {}
    for path in sorted(image_dir.glob("img*.jpg"), key=lambda p: int(p.stem[3:])):
        frame = cv2.imread(str(path))
        for left, top, w, h in ignored:
            x1, y1 = max(0, int(left)), max(0, int(top))
            x2, y2 = min(frame.shape[1], int(left + w) + 1), min(frame.shape[0], int(top + h) + 1)
            frame[y1:y2, x1:x2] = GREY

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf,
            imgsz=imgsz,
            agnostic_nms=True,
            device=0,
            verbose=False,
        )[0]

        boxes = result.boxes
        frames[int(path.stem[3:])] = [] if boxes.id is None else [
            (int(tid), *(round(v, 1) for v in xyxy)) for xyxy, tid in zip(boxes.xyxy.tolist(), boxes.id.tolist())
        ]
    return frames


def load_or_track(model_holder, name, ignored, args):
    """Tracked boxes for one sequence: from the cache if it was made with the same weights and
    settings, otherwise by running the tracker (which refreshes the cache)."""
    cache = TRACKS / f"{name}.json"
    settings = {"weights": Path(args.weights).resolve().as_posix(), "conf": args.conf, "imgsz": args.imgsz}
    if cache.exists():
        data = json.loads(cache.read_text())
        if data["settings"] == settings:
            return {int(k): [tuple(b) for b in v] for k, v in data["frames"].items()}
    if "model" not in model_holder:
        model_holder["model"] = YOLO(str(args.weights))
    frames = run_tracker(model_holder["model"], IMAGES / name, ignored, args.conf, args.imgsz)
    TRACKS.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"settings": settings, "frames": frames}))
    return frames


def tracks_from_frames(frames):
    """track id -> [(frame number, bottom-centre x, bottom-centre y, box height)], sorted by frame."""
    tracks = defaultdict(list)
    for number in sorted(frames):
        for tid, x1, y1, x2, y2 in frames[number]:
            tracks[tid].append((number, (x1 + x2) / 2, y2, y2 - y1))
    return tracks


def axis_deg(dx, dy):
    """Direction of travel folded to 0..180 degrees, so opposite lanes share an axis."""
    return math.degrees(math.atan2(dy, dx)) % 180


def axis_gap(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)


def in_ignored(x, y, ignored):
    return any(left <= x <= left + w and top <= y <= top + h for left, top, w, h in ignored)


def place_line(tracks, ignored, width, height):
    """Place a counting segment across the main flow of the predicted tracks (see module docstring)."""
    moving = {}
    for tid, pts in tracks.items():
        if len(pts) < MIN_TRACK_FRAMES:
            continue
        dx, dy = pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]
        travel = math.hypot(dx, dy)
        if travel >= MIN_TRAVEL:
            moving[tid] = (axis_deg(dx, dy), travel)
    if len(moving) < MIN_FLOW_TRACKS:
        return {"suitable": False, "reason": f"only {len(moving)} moving tracks", "p1": None, "p2": None}

    # histogram of axes, each bin summed with its neighbours (wrapping at 180) to find the densest axis
    bins = 180 // AXIS_BIN_DEG
    hist = np.zeros(bins)
    for angle, _ in moving.values():
        hist[int(angle // AXIS_BIN_DEG) % bins] += 1
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)
    peak = (int(np.argmax(smooth)) + 0.5) * AXIS_BIN_DEG
    members = [tid for tid, (angle, _) in moving.items() if axis_gap(angle, peak) <= AXIS_TOLERANCE_DEG]
    # refine the axis as the travel-weighted mean of the members (doubled angles handle the 0/180 wrap)
    c = sum(moving[t][1] * math.cos(math.radians(2 * moving[t][0])) for t in members)
    s = sum(moving[t][1] * math.sin(math.radians(2 * moving[t][0])) for t in members)
    axis = math.degrees(math.atan2(s, c)) / 2 % 180
    share = len(members) / len(moving)

    u = np.array([math.cos(math.radians(axis)), math.sin(math.radians(axis))])  # along the traffic
    n = np.array([-u[1], u[0]])  # across the traffic: the line's direction
    member_pts = {t: np.array([(p[1], p[2], p[3]) for p in tracks[t]]) for t in members}
    along = {t: pts[:, :2] @ u for t, pts in member_pts.items()}
    all_along = np.concatenate(list(along.values()))
    # middle of the flow only, so tracks are seen for a while on both sides of the line
    lo, hi = np.percentile(all_along, [20, 80])

    candidates = []
    for pos in np.linspace(lo, hi, 41):
        covering = [t for t in members if along[t].min() < pos < along[t].max()]
        if not covering:
            continue
        # bottom-centre points of covering tracks near this position: they give the flow's width here
        near = np.concatenate([member_pts[t][np.abs(along[t] - pos) < 15] for t in covering])
        if len(near) == 0:
            continue
        across = near[:, :2] @ n
        box_height = float(np.median(near[:, 2]))
        a_lo, a_hi = float(np.percentile(across, 2)) - box_height / 2, float(np.percentile(across, 98)) + box_height / 2
        p1, p2 = pos * u + a_lo * n, pos * u + a_hi * n
        clear = not any(in_ignored(*(p1 + (p2 - p1) * k / 20), ignored) for k in range(21))
        inside = all(EDGE_MARGIN <= p[0] <= width - EDGE_MARGIN and EDGE_MARGIN <= p[1] <= height - EDGE_MARGIN
                     for p in (p1, p2))
        candidates.append({"covering": len(covering), "box_height": box_height, "clear": clear, "inside": inside,
                           "p1": p1, "p2": p2})
    if not candidates:
        return {"suitable": False, "reason": "no line position found", "p1": None, "p2": None}

    # positions crossed by at least 80% of the best coverage; of those, prefer ones clear of ignored
    # regions and inside the frame, then the one nearest the camera (biggest boxes)
    max_cover = max(c["covering"] for c in candidates)
    good = [c for c in candidates if c["covering"] >= 0.8 * max_cover]
    pool = ([c for c in good if c["clear"] and c["inside"]] or [c for c in good if c["clear"]]
            or [c for c in good if c["inside"]] or good)
    best = max(pool, key=lambda c: c["box_height"])
    clear, covering, p1, p2 = best["clear"], best["covering"], best["p1"], best["p2"]
    p1 = [float(np.clip(p1[0], 0, width - 1)), float(np.clip(p1[1], 0, height - 1))]
    p2 = [float(np.clip(p2[0], 0, width - 1)), float(np.clip(p2[1], 0, height - 1))]
    reasons = []
    if share < MIN_FLOW_SHARE:
        reasons.append(f"main flow is only {share:.0%} of moving tracks")
    if not clear:
        reasons.append("line touches an ignored region")
    return {
        "suitable": not reasons,
        "reason": "; ".join(reasons),
        "p1": p1,
        "p2": p2,
        "axis_deg": round(axis, 1),
        "main_flow_share": round(share, 3),
        "moving_tracks": len(moving),
        "main_flow_tracks": len(members),
        "tracks_crossing_line": covering,
        "members": members,
    }


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(a, b, p1, p2):
    return (cross(p1, p2, a) * cross(p1, p2, b) < 0) and (cross(a, b, p1) * cross(a, b, p2) < 0)


def count_crossings(tracks, p1, p2, min_side_frames):
    """Number of tracks whose bottom-centre path crosses segment p1-p2, each counted at most once."""
    count = 0
    for pts in tracks.values():
        path = [(p[1], p[2]) for p in pts]
        for k in range(1, len(path)):
            if segments_intersect(path[k - 1], path[k], p1, p2):
                if k >= min_side_frames and len(path) - k >= min_side_frames:
                    count += 1
                break
    return count


def iou_matrix(a, b):
    """IoU between every box in a and every box in b; boxes are (x1, y1, x2, y2) rows."""
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def box_match(targets, frames, ignored, min_iou):
    """Match tracked boxes to GT boxes frame by frame; return (matched, extra, missed, id switches)."""
    matched = extra = missed = switches = 0
    last_track = {}
    for number in sorted(set(targets) | set(frames)):
        gts = targets.get(number, [])
        # a tracked box centred in an ignored region has no GT to match by design; leave it out
        preds = [p for p in frames.get(number, []) if not in_ignored((p[1] + p[3]) / 2, (p[2] + p[4]) / 2, ignored)]
        pairs = []
        if gts and preds:
            ious = iou_matrix(np.array([g[1:] for g in gts]), np.array([p[1:] for p in preds]))
            rows, cols = linear_sum_assignment(-ious)
            pairs = [(r, c) for r, c in zip(rows, cols) if ious[r, c] >= min_iou]
        for r, c in pairs:
            gid, pid = gts[r][0], preds[c][0]
            if gid in last_track and last_track[gid] != pid:
                switches += 1
            last_track[gid] = pid
        matched += len(pairs)
        missed += len(gts) - len(pairs)
        extra += len(preds) - len(pairs)
    return matched, extra, missed, switches


def draw_preview(name, weather, targets, ignored, tracks, line, row):
    """Middle frame with ignored regions (red), GT boxes (green), predicted tracks (cyan = main flow,
    grey = other) and the counting line (yellow, or red when the sequence is unsuitable)."""
    number = sorted(targets)[len(targets) // 2]
    image = cv2.imread(str(IMAGES / name / f"img{number:05d}.jpg"))
    shade = image.copy()
    for left, top, w, h in ignored:
        cv2.rectangle(shade, (int(left), int(top)), (int(left + w), int(top + h)), (0, 0, 255), -1)
    image = cv2.addWeighted(shade, 0.35, image, 0.65, 0)
    members = set(line.get("members", []))
    for tid, pts in tracks.items():
        if len(pts) < MIN_TRACK_FRAMES:
            continue
        path = np.array([(p[1], p[2]) for p in pts], dtype=np.int32)
        cv2.polylines(image, [path], False, (255, 255, 0) if tid in members else (160, 160, 160), 1)
    for _, x1, y1, x2, y2 in targets[number]:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    if line["p1"]:
        color = (0, 255, 255) if line["suitable"] else (0, 0, 255)
        cv2.line(image, tuple(int(v) for v in line["p1"]), tuple(int(v) for v in line["p2"]), color, 3)
    text = f"{name} ({weather}) gt {row['gt_count']} pred {row['pred_count']} err {row['error']:+d}"
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if not line["suitable"]:
        cv2.putText(image, f"unsuitable: {line['reason']}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(LINES / f"{name}.jpg"), image)


def summarize(rows):
    n = len(rows)
    out = {
        "sequences": n,
        "mae": sum(r["abs_error"] for r in rows) / n,
        "bias": sum(r["error"] for r in rows) / n,
        "within_1": sum(r["abs_error"] <= 1 for r in rows) / n,
        "within_2": sum(r["abs_error"] <= 2 for r in rows) / n,
    }
    for level in IOU_LEVELS:
        tag = f"{int(level * 100)}"
        m = sum(r[f"matched_{tag}"] for r in rows)
        e = sum(r[f"extra_{tag}"] for r in rows)
        x = sum(r[f"missed_{tag}"] for r in rows)
        out[f"precision_{tag}"] = m / max(m + e, 1)
        out[f"recall_{tag}"] = m / max(m + x, 1)
    out["id_switches_per_100_vehicles"] = 100 * sum(r["id_switches"] for r in rows) / max(sum(r["gt_vehicles"] for r in rows), 1)
    return out


def evaluate(args):
    LINES.mkdir(parents=True, exist_ok=True)
    model_holder = {}
    rows = []

    xml_paths = sorted(TEST_XML.glob("*.xml"))
    if args.limit:
        xml_paths = xml_paths[:args.limit]
    for i, xml_path in enumerate(xml_paths, 1):
        name = xml_path.stem
        weather, ignored, targets = parse_test_sequence(xml_path)
        frames = load_or_track(model_holder, name, ignored, args)
        height, width = cv2.imread(str(next((IMAGES / name).glob("img*.jpg")))).shape[:2]

        pred_tracks = tracks_from_frames(frames)
        gt_tracks = tracks_from_frames(targets)
        line = place_line(pred_tracks, ignored, width, height)
        (LINES / f"{name}.json").write_text(json.dumps({k: v for k, v in line.items() if k != "members"}, indent=2))

        if line["p1"]:
            gt_count = count_crossings(gt_tracks, line["p1"], line["p2"], args.min_side_frames)
            pred_count = count_crossings(pred_tracks, line["p1"], line["p2"], args.min_side_frames)
        else:
            gt_count = pred_count = 0
        row = {
            "sequence": name, "weather": weather, "frames": len(targets),
            "suitable": line["suitable"], "reason": line["reason"],
            "gt_count": gt_count, "pred_count": pred_count,
            "error": pred_count - gt_count, "abs_error": abs(pred_count - gt_count),
            "gt_vehicles": len(gt_tracks), "pred_tracks": len(pred_tracks),
        }
        for level in IOU_LEVELS:
            tag = f"{int(level * 100)}"
            matched, extra, missed, switches = box_match(targets, frames, ignored, level)
            row.update({f"matched_{tag}": matched, f"extra_{tag}": extra, f"missed_{tag}": missed})
            if level == IOU_LEVELS[0]:
                row["id_switches"] = switches
        rows.append(row)
        draw_preview(name, weather, targets, ignored, pred_tracks, line, row)

        flag = "" if line["suitable"] else f"  UNSUITABLE ({line['reason']})"
        print(f"[{i}/{len(xml_paths)}] {name} ({weather}) gt {gt_count:3d} pred {pred_count:3d} err {row['error']:+3d}"
              f" | P50 {row['matched_50'] / max(row['matched_50'] + row['extra_50'], 1):.3f}"
              f" R50 {row['matched_50'] / max(row['matched_50'] + row['missed_50'], 1):.3f}"
              f" id switches {row['id_switches']}{flag}", flush=True)

    with open(OUT / "per_sequence.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups = [(w, [r for r in rows if r["weather"] == w]) for w in sorted({r["weather"] for r in rows})]
    groups += [("all", rows), ("suitable", [r for r in rows if r["suitable"]]),
               ("unsuitable", [r for r in rows if not r["suitable"]])]
    lines = [f"{'group':10s} {'seqs':>4s} {'MAE':>5s} {'bias':>6s} {'±1':>5s} {'±2':>5s} | "
             f"{'P50':>5s} {'R50':>5s} {'P70':>5s} {'R70':>5s} | {'IDsw/100':>8s}"]
    with open(OUT / "summary.csv", "w", newline="") as f:
        writer = None
        for label, group in groups:
            if not group:
                continue
            s = summarize(group)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=["group", *s.keys()])
                writer.writeheader()
            writer.writerow({"group": label, **{k: round(v, 4) for k, v in s.items()}})
            lines.append(f"{label:10s} {s['sequences']:4d} {s['mae']:5.2f} {s['bias']:+6.2f} {s['within_1']:5.0%} "
                         f"{s['within_2']:5.0%} | {s['precision_50']:5.3f} {s['recall_50']:5.3f} "
                         f"{s['precision_70']:5.3f} {s['recall_70']:5.3f} | {s['id_switches_per_100_vehicles']:8.1f}")

    print("\n" + "\n".join(lines))
    print(f"\nper-sequence: {OUT / 'per_sequence.csv'}")
    print(f"summary: {OUT / 'summary.csv'}")
    print(f"line images and line files: {LINES}")
