"""Count vehicles crossing a line on the 40 held-out UA-DETRAC Test sequences and compare
against XML ground truth, per weather condition.

Tracking: the given weights run with a tracker (ByteTrack by default, or a yaml from
pipeline/trackers/) over every frame, never subsampled, since tracking needs continuous motion.
Ignored regions are greyed out first, same as yolo_export.py, since the annotators left vehicles
there unlabelled. Every tracker setup gets its own folder, data/runs/count-eval/<tag>/, with the
tracked boxes cached in tracks/, so changing the line or the counting rule re-scores without the GPU.

Line: placed automatically from the predicted tracks, never from the ground truth, so the same
method works on game footage without annotations. It is the segment that the most moving vehicles
cross: every line angle (5 deg steps) and position is tried, and a moving track (seen >= 1 s,
moved >= MIN_TRAVEL px) counts for a candidate if it crosses it at more than MIN_CROSS_ANGLE_DEG.
Among candidates crossed by at least 90% of the best one's tracks, lines clear of ignored regions
and inside the frame win, then the one nearest the camera (biggest boxes). The segment spans where
the tracks actually cross. This does not depend on perspective: lanes that fan out towards a
vanishing point are all crossed by one line. If even the best line is crossed by less than
MIN_LINE_SHARE of the moving tracks (e.g. an intersection), the sequence is flagged unsuitable.

Counting: the reference point of a box is at --ref-point of its height, measured from the top
(1.0 = bottom edge, 0.5 = centre), horizontally centred. A track counts once, the first time
it moves from one side of the segment to the other within the segment's length. With
--min-side-frames N it must also be seen N frames on each side (off by default: with fragmented
tracks it rejects real vehicles). The ground truth is counted on the same line with the same rule.

Crossing pairing: every counted crossing is paired with a GT crossing of the vehicle under the
track (Hungarian match at IoU >= PAIR_IOU near that frame) within PAIR_FRAMES frames. Unpaired
counted crossings are split into: the same vehicle counted again, no labelled vehicle there, or a
real vehicle whose GT box does not cross the line (the predicted and GT boxes disagree near the
line). Unpaired GT crossings are missed vehicles.

Clip windows: each sequence is cut into consecutive CLIP_FRAMES windows (30 s, the game's clip
length; the remainder is dropped) and the crossings are counted per window, as a game clip would
be. Tracking still runs over the whole sequence, so a window does not see the tracker warming up
at its start the way a separately processed clip would.

Box score: tracked boxes are matched to GT boxes frame by frame with the Hungarian method on IoU,
at IoU 0.5 and 0.7: matched, extra (no GT) and missed (no prediction) boxes, and ID switches, i.e.
a GT vehicle whose matched track id changes, which a line counter sees as a new vehicle. The same
score is repeated for the band around the line only (BAND_FRACTION of the flow's length on each
side, within the line's width), where a broken track actually changes the count, and recall is
split by GT box height, since far-away vehicles are small.
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
TRACKER_DIR = Path(__file__).resolve().parent / "trackers"
GREY = (114, 114, 114)

MIN_TRACK_FRAMES = 25  # 1 s at 25 fps
MIN_TRAVEL = 40.0  # px; slower tracks are parked or false
LINE_ANGLE_STEP = 5  # deg
LINE_POSITIONS = 31  # per angle, between the 20th and 80th percentile of the tracks
MIN_CROSS_ANGLE_DEG = 30  # a track running almost along the line does not count as crossing it
COVER_KEEP = 0.9  # candidates crossed by at least this share of the best candidate's tracks
MIN_LINE_SHARE = 0.7
MIN_FLOW_TRACKS = 5
EDGE_MARGIN = 10  # px; a line this close to the frame edge meets vehicles that are cut off or leaving
BAND_FRACTION = 0.25
IOU_LEVELS = (0.5, 0.7)
SIZE_BUCKETS = (("lt25", 0, 25), ("25to50", 25, 50), ("50to100", 50, 100), ("ge100", 100, math.inf))
PAIR_IOU = 0.3
PAIR_FRAMES = 10
CLIP_FRAMES = 750  # 30 s at 25 fps


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


def tracker_path(name):
    """A yaml in pipeline/trackers/ by name, otherwise an ultralytics built-in such as bytetrack.yaml."""
    local = TRACKER_DIR / name
    return str(local) if local.exists() else name


def run_tracker(model, image_dir, ignored, args):
    """Run the tracker over every frame in image_dir; return {frame number: [(track id, x1, y1, x2, y2)]}."""
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
            tracker=tracker_path(args.tracker),
            conf=args.conf,
            imgsz=args.imgsz,
            agnostic_nms=True,
            device=0,
            verbose=False,
        )[0]

        boxes = result.boxes
        frames[int(path.stem[3:])] = [] if boxes.id is None else [
            (int(tid), *(round(v, 1) for v in xyxy)) for xyxy, tid in zip(boxes.xyxy.tolist(), boxes.id.tolist())
        ]
    return frames


def load_or_track(model_holder, name, ignored, args, tracks_dir):
    """Tracked boxes for one sequence: from the cache if it was made with the same weights and
    settings, otherwise by running the tracker (which refreshes the cache)."""
    cache = tracks_dir / f"{name}.json"
    settings = {"weights": Path(args.weights).resolve().as_posix(), "conf": args.conf, "imgsz": args.imgsz,
                "tracker": args.tracker}
    if cache.exists():
        data = json.loads(cache.read_text())
        if data["settings"] == settings:
            return {int(k): [tuple(b) for b in v] for k, v in data["frames"].items()}
    if "model" not in model_holder:
        model_holder["model"] = YOLO(str(args.weights))
    frames = run_tracker(model_holder["model"], IMAGES / name, ignored, args)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"settings": settings, "frames": frames}))
    return frames


def ref_point(box, ref):
    """Counting point of an (x1, y1, x2, y2) box: horizontally centred, ref of the height down from the top."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y1 + ref * (y2 - y1)


def tracks_from_frames(frames, ref):
    """track id -> [(frame number, reference point x, reference point y, box height)], sorted by frame."""
    tracks = defaultdict(list)
    for number in sorted(frames):
        for tid, *box in frames[number]:
            tracks[tid].append((number, *ref_point(box, ref), box[3] - box[1]))
    return tracks


def in_ignored(x, y, ignored):
    return any(left <= x <= left + w and top <= y <= top + h for left, top, w, h in ignored)


def unsuitable(reason):
    return {"suitable": False, "reason": reason, "p1": None, "p2": None, "members": []}


def place_line(tracks, ignored, width, height):
    """Find the segment crossed by the most moving tracks (see module docstring)."""
    moving = {}
    for tid, pts in tracks.items():
        if len(pts) < MIN_TRACK_FRAMES:
            continue
        arr = np.array([(p[1], p[2], p[3]) for p in pts])
        move = arr[-1, :2] - arr[0, :2]
        travel = float(np.hypot(*move))
        if travel >= MIN_TRAVEL:
            moving[tid] = (arr, move / travel)
    if len(moving) < MIN_FLOW_TRACKS:
        return unsuitable(f"only {len(moving)} moving tracks")

    all_pts = np.concatenate([arr[:, :2] for arr, _ in moving.values()])
    min_dot = math.sin(math.radians(MIN_CROSS_ANGLE_DEG))  # |direction . normal| for a crossing angle
    candidates = []
    for angle in range(0, 180, LINE_ANGLE_STEP):
        along = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        normal = np.array([-along[1], along[0]])
        steep = []
        for tid, (arr, direction) in moving.items():
            if abs(direction @ normal) >= min_dot:
                q = arr[:, :2] @ normal
                steep.append((tid, arr, q, q.min(), q.max()))
        if len(steep) < MIN_FLOW_TRACKS:
            continue
        # middle of the flow only: near the frame edge vehicles are cut off and tracks start or end
        lo, hi = np.percentile(all_pts @ normal, [20, 80])
        for pos in np.linspace(lo, hi, LINE_POSITIONS):
            hits = []  # (track id, position along the line where it crosses, box height there)
            for tid, arr, q, q_min, q_max in steep:
                if not q_min < pos < q_max:
                    continue
                side = q > pos
                k = int(np.nonzero(side[1:] != side[:-1])[0][0]) + 1
                t = (pos - q[k - 1]) / (q[k] - q[k - 1])
                point = arr[k - 1, :2] + t * (arr[k, :2] - arr[k - 1, :2])
                hits.append((tid, float(point @ along), float(arr[k, 2])))
            if len(hits) < MIN_FLOW_TRACKS:
                continue
            r = np.array([h[1] for h in hits])
            box_height = float(np.median([h[2] for h in hits]))
            r_lo = float(np.percentile(r, 2)) - box_height / 2
            r_hi = float(np.percentile(r, 98)) + box_height / 2
            p1, p2 = pos * normal + r_lo * along, pos * normal + r_hi * along
            clear = not any(in_ignored(*(p1 + (p2 - p1) * k / 20), ignored) for k in range(21))
            inside = all(EDGE_MARGIN <= p[0] <= width - EDGE_MARGIN and EDGE_MARGIN <= p[1] <= height - EDGE_MARGIN
                         for p in (p1, p2))
            candidates.append({"covering": len(hits), "box_height": box_height, "clear": clear, "inside": inside,
                               "p1": p1, "p2": p2, "angle": angle, "pos": float(pos), "r_lo": r_lo, "r_hi": r_hi,
                               "members": [h[0] for h in hits]})
    if not candidates:
        return unsuitable("no line crossed by enough moving tracks")

    max_cover = max(c["covering"] for c in candidates)
    good = [c for c in candidates if c["covering"] >= COVER_KEEP * max_cover]
    pool = ([c for c in good if c["clear"] and c["inside"]] or [c for c in good if c["clear"]]
            or [c for c in good if c["inside"]] or good)
    best = max(pool, key=lambda c: c["box_height"])

    # the band: BAND_FRACTION of the flow's length (along the normal) on each side of the line
    normal = np.array([-math.sin(math.radians(best["angle"])), math.cos(math.radians(best["angle"]))])
    member_pts = np.concatenate([moving[t][0][:, :2] for t in best["members"]])
    q_lo, q_hi = np.percentile(member_pts @ normal, [5, 95])
    share = best["covering"] / len(moving)
    reasons = []
    if share < MIN_LINE_SHARE:
        reasons.append(f"best line is crossed by only {share:.0%} of moving tracks")
    if not best["clear"]:
        reasons.append("line touches an ignored region")
    return {
        "suitable": not reasons,
        "reason": "; ".join(reasons),
        "p1": [float(np.clip(best["p1"][0], 0, width - 1)), float(np.clip(best["p1"][1], 0, height - 1))],
        "p2": [float(np.clip(best["p2"][0], 0, width - 1)), float(np.clip(best["p2"][1], 0, height - 1))],
        "angle_deg": best["angle"],
        "pos": best["pos"],
        "r_lo": best["r_lo"],
        "r_hi": best["r_hi"],
        "band_half": float(BAND_FRACTION * (q_hi - q_lo)),
        "line_share": round(share, 3),
        "moving_tracks": len(moving),
        "tracks_crossing_line": best["covering"],
        "members": best["members"],
    }


def band_test(line):
    """(x, y) -> True if the point is in the band around the line; None when there is no line."""
    if not line["p1"]:
        return None
    angle = math.radians(line["angle_deg"])
    along = (math.cos(angle), math.sin(angle))
    normal = (-along[1], along[0])

    def inside(x, y):
        q = x * normal[0] + y * normal[1]
        r = x * along[0] + y * along[1]
        return abs(q - line["pos"]) <= line["band_half"] and line["r_lo"] <= r <= line["r_hi"]
    return inside


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(a, b, p1, p2):
    return (cross(p1, p2, a) * cross(p1, p2, b) < 0) and (cross(a, b, p1) * cross(a, b, p2) < 0)


def crossing_frames(tracks, p1, p2, min_side_frames):
    """track id -> frame number of its first crossing of segment p1-p2, for the tracks that count."""
    crossings = {}
    for tid, pts in tracks.items():
        for k in range(1, len(pts)):
            if segments_intersect(pts[k - 1][1:3], pts[k][1:3], p1, p2):
                if k >= min_side_frames and len(pts) - k >= min_side_frames:
                    crossings[tid] = pts[k][0]
                break
    return crossings


def match_map(targets, frames, min_iou):
    """(frame number, track id) -> GT id, from a Hungarian match of every frame at IoU >= min_iou."""
    matches = {}
    for number in set(targets) & set(frames):
        gts, preds = targets[number], frames[number]
        if not gts or not preds:
            continue
        ious = iou_matrix(np.array([g[1:] for g in gts]), np.array([p[1:] for p in preds]))
        for r, col in zip(*linear_sum_assignment(-ious)):
            if ious[r, col] >= min_iou:
                matches[(number, preds[col][0])] = gts[r][0]
    return matches


def pair_crossings(gt_cross, pred_cross, matches):
    """Pair counted crossings with GT crossings (see module docstring); return a dict of counters."""
    c = defaultdict(int)
    c["gt_crossings"] = len(gt_cross)
    offsets = sorted(range(-PAIR_FRAMES, PAIR_FRAMES + 1), key=abs)
    paired = set()
    for tid, frame in sorted(pred_cross.items(), key=lambda item: item[1]):
        gid = next((matches[(frame + d, tid)] for d in offsets if (frame + d, tid) in matches), None)
        if gid is None:
            c["extra_no_vehicle"] += 1
        elif gid in paired:
            c["extra_same_vehicle"] += 1
        elif gid not in gt_cross or abs(gt_cross[gid] - frame) > PAIR_FRAMES:
            c["extra_gt_not_crossing"] += 1
        else:
            c["correct"] += 1
            paired.add(gid)
    c["missed"] = len(gt_cross) - len(paired)
    return c


def window_counts(gt_cross, pred_cross, frame_numbers):
    """[(first frame, GT count, predicted count)] per consecutive CLIP_FRAMES window of the sequence."""
    start, end = min(frame_numbers), max(frame_numbers)
    windows = []
    for first in range(start, end - CLIP_FRAMES + 2, CLIP_FRAMES):
        inside = lambda frame: first <= frame < first + CLIP_FRAMES
        windows.append((first, sum(map(inside, gt_cross.values())), sum(map(inside, pred_cross.values()))))
    return windows


def iou_matrix(a, b):
    """IoU between every box in a and every box in b; boxes are (x1, y1, x2, y2) rows."""
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def size_bucket(box):
    height = box[3] - box[1]
    return next(name for name, lo, hi in SIZE_BUCKETS if lo <= height < hi)


def box_match(targets, frames, ignored, min_iou, in_band, ref):
    """Match tracked boxes to GT boxes frame by frame; return a dict of counters for the whole frame,
    for the band around the line (points tested at the box's reference point) and per GT box height."""
    c = defaultdict(int)
    last_track, last_band_track = {}, {}
    band_vehicles = set()
    for number in sorted(set(targets) | set(frames)):
        gts = targets.get(number, [])
        # a tracked box centred in an ignored region has no GT to match by design; leave it out
        preds = [p for p in frames.get(number, []) if not in_ignored((p[1] + p[3]) / 2, (p[2] + p[4]) / 2, ignored)]
        pairs = []
        if gts and preds:
            ious = iou_matrix(np.array([g[1:] for g in gts]), np.array([p[1:] for p in preds]))
            rows, cols = linear_sum_assignment(-ious)
            pairs = [(r, col) for r, col in zip(rows, cols) if ious[r, col] >= min_iou]
        matched_gt = {r: col for r, col in pairs}
        matched_pred = {col for _, col in pairs}

        c["matched"] += len(pairs)
        c["missed"] += len(gts) - len(pairs)
        c["extra"] += len(preds) - len(pairs)
        for r, g in enumerate(gts):
            bucket = size_bucket(g[1:])
            c[f"gt_{bucket}"] += 1
            c[f"matched_{bucket}"] += r in matched_gt
            gid = g[0]
            g_in_band = in_band is not None and in_band(*ref_point(g[1:], ref))
            if g_in_band:
                band_vehicles.add(gid)
                c["band_gt"] += 1
                c["band_gt_matched"] += r in matched_gt
            if r in matched_gt:
                pid = preds[matched_gt[r]][0]
                if gid in last_track and last_track[gid] != pid:
                    c["switches"] += 1
                last_track[gid] = pid
                if g_in_band:
                    if gid in last_band_track and last_band_track[gid] != pid:
                        c["band_switches"] += 1
                    last_band_track[gid] = pid
        if in_band is not None:
            for col, p in enumerate(preds):
                if in_band(*ref_point(p[1:], ref)):
                    c["band_pred"] += 1
                    c["band_pred_matched"] += col in matched_pred
    c["band_vehicles"] = len(band_vehicles)
    return c


def draw_preview(path, name, weather, targets, ignored, tracks, line, row):
    """Middle frame with ignored regions (red), GT boxes (green), predicted tracks (cyan = crosses the
    line, grey = other), the band (thin yellow) and the counting line (thick yellow, red if unsuitable)."""
    number = sorted(targets)[len(targets) // 2]
    image = cv2.imread(str(IMAGES / name / f"img{number:05d}.jpg"))
    shade = image.copy()
    for left, top, w, h in ignored:
        cv2.rectangle(shade, (int(left), int(top)), (int(left + w), int(top + h)), (0, 0, 255), -1)
    image = cv2.addWeighted(shade, 0.35, image, 0.65, 0)
    members = set(line["members"])
    for tid, pts in tracks.items():
        if len(pts) < MIN_TRACK_FRAMES:
            continue
        points = np.array([(p[1], p[2]) for p in pts], dtype=np.int32)
        cv2.polylines(image, [points], False, (255, 255, 0) if tid in members else (160, 160, 160), 1)
    for _, x1, y1, x2, y2 in targets[number]:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    if line["p1"]:
        color = (0, 255, 255) if line["suitable"] else (0, 0, 255)
        angle = math.radians(line["angle_deg"])
        along = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-along[1], along[0]])
        for offset in (-line["band_half"], line["band_half"]):
            q = line["pos"] + offset
            a, b = q * normal + line["r_lo"] * along, q * normal + line["r_hi"] * along
            cv2.line(image, tuple(int(v) for v in a), tuple(int(v) for v in b), color, 1)
        cv2.line(image, tuple(int(v) for v in line["p1"]), tuple(int(v) for v in line["p2"]), color, 3)
    text = f"{name} ({weather}) gt {row['gt_count']} pred {row['pred_count']} err {row['error']:+d}"
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if not line["suitable"]:
        cv2.putText(image, f"unsuitable: {line['reason']}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(path), image)


def ratio(a, b):
    return a / b if b else float("nan")


PAIR_KEYS = ("correct", "missed", "extra_same_vehicle", "extra_no_vehicle", "extra_gt_not_crossing")


def summarize(rows, windows):
    total = lambda key: sum(r[key] for r in rows)
    n = len(rows)
    out = {
        "sequences": n,
        "mae": total("abs_error") / n,
        "bias": total("error") / n,
        "within_1": sum(r["abs_error"] <= 1 for r in rows) / n,
        "within_2": sum(r["abs_error"] <= 2 for r in rows) / n,
    }
    for level in IOU_LEVELS:
        tag = f"{int(level * 100)}"
        m, e, x = total(f"matched_{tag}"), total(f"extra_{tag}"), total(f"missed_{tag}")
        out[f"precision_{tag}"] = ratio(m, m + e)
        out[f"recall_{tag}"] = ratio(m, m + x)
    out["id_switches_per_100_vehicles"] = 100 * ratio(total("id_switches"), total("gt_vehicles"))
    out["band_precision_50"] = ratio(total("band_pred_matched"), total("band_pred"))
    out["band_recall_50"] = ratio(total("band_gt_matched"), total("band_gt"))
    out["band_id_switches_per_100_vehicles"] = 100 * ratio(total("band_switches"), total("band_vehicles"))
    for bucket, _, _ in SIZE_BUCKETS:
        out[f"recall_50_{bucket}"] = ratio(total(f"matched_{bucket}"), total(f"gt_{bucket}"))
        out[f"gt_boxes_{bucket}"] = total(f"gt_{bucket}")
    # crossing pairing, as shares of the GT crossings
    out["gt_crossings"] = total("gt_crossings")
    for key in PAIR_KEYS:
        out[f"crossings_{key}"] = ratio(total(key), out["gt_crossings"])
    # 30 s windows
    errors = np.array([w["error"] for w in windows]) if windows else np.array([np.nan])
    out["clips"] = len(windows)
    out["clip_mean_gt_count"] = float(np.mean([w["gt_count"] for w in windows])) if windows else float("nan")
    out["clip_exact"] = float(np.mean(errors == 0))
    out["clip_within_1"] = float(np.mean(np.abs(errors) <= 1))
    out["clip_within_2"] = float(np.mean(np.abs(errors) <= 2))
    out["clip_mae"] = float(np.mean(np.abs(errors)))
    out["clip_bias"] = float(np.mean(errors))
    return out


def evaluate(args):
    tag = args.tag or f"{Path(args.tracker).stem}-conf{args.conf:g}-img{args.imgsz}"
    out_dir = RUNS / "count-eval" / tag
    lines_dir = out_dir / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    model_holder = {}
    rows, windows = [], []

    xml_paths = sorted(TEST_XML.glob("*.xml"))
    if args.limit:
        xml_paths = xml_paths[:args.limit]
    for i, xml_path in enumerate(xml_paths, 1):
        name = xml_path.stem
        weather, ignored, targets = parse_test_sequence(xml_path)
        frames = load_or_track(model_holder, name, ignored, args, out_dir / "tracks")
        height, width = cv2.imread(str(next((IMAGES / name).glob("img*.jpg")))).shape[:2]

        pred_tracks = tracks_from_frames(frames, args.ref_point)
        gt_tracks = tracks_from_frames(targets, args.ref_point)
        line = place_line(pred_tracks, ignored, width, height)
        (lines_dir / f"{name}.json").write_text(json.dumps({k: v for k, v in line.items() if k != "members"}, indent=2))

        if line["p1"]:
            gt_cross = crossing_frames(gt_tracks, line["p1"], line["p2"], args.min_side_frames)
            pred_cross = crossing_frames(pred_tracks, line["p1"], line["p2"], args.min_side_frames)
        else:
            gt_cross, pred_cross = {}, {}
        gt_count, pred_count = len(gt_cross), len(pred_cross)
        row = {
            "sequence": name, "weather": weather, "frames": len(targets),
            "suitable": line["suitable"], "reason": line["reason"],
            "gt_count": gt_count, "pred_count": pred_count,
            "error": pred_count - gt_count, "abs_error": abs(pred_count - gt_count),
            "gt_vehicles": len(gt_tracks), "pred_tracks": len(pred_tracks),
        }
        in_band = band_test(line)
        for level in IOU_LEVELS:
            level_tag = f"{int(level * 100)}"
            c = box_match(targets, frames, ignored, level, in_band, args.ref_point)
            row.update({f"matched_{level_tag}": c["matched"], f"extra_{level_tag}": c["extra"],
                        f"missed_{level_tag}": c["missed"]})
            if level == IOU_LEVELS[0]:
                row["id_switches"] = c["switches"]
                for key in ("band_gt", "band_gt_matched", "band_pred", "band_pred_matched", "band_switches",
                            "band_vehicles"):
                    row[key] = c[key]
                for bucket, _, _ in SIZE_BUCKETS:
                    row[f"gt_{bucket}"] = c[f"gt_{bucket}"]
                    row[f"matched_{bucket}"] = c[f"matched_{bucket}"]
        pairing = pair_crossings(gt_cross, pred_cross, match_map(targets, frames, PAIR_IOU))
        row.update({"gt_crossings": pairing["gt_crossings"], **{key: pairing[key] for key in PAIR_KEYS}})
        for first, gt_n, pred_n in window_counts(gt_cross, pred_cross, list(targets)):
            windows.append({"sequence": name, "weather": weather, "suitable": line["suitable"], "first_frame": first,
                            "gt_count": gt_n, "pred_count": pred_n, "error": pred_n - gt_n})
        rows.append(row)
        draw_preview(lines_dir / f"{name}.jpg", name, weather, targets, ignored, pred_tracks, line, row)

        flag = "" if line["suitable"] else f"  UNSUITABLE ({line['reason']})"
        print(f"[{i}/{len(xml_paths)}] {name} ({weather}) gt {gt_count:3d} pred {pred_count:3d} err {row['error']:+3d}"
              f" | crossings correct {pairing['correct']} missed {pairing['missed']}"
              f" extra {pairing['extra_same_vehicle'] + pairing['extra_no_vehicle'] + pairing['extra_gt_not_crossing']}"
              f" | id switches {row['id_switches']} (band {row['band_switches']}){flag}", flush=True)

    for filename, data in (("per_sequence.csv", rows), ("clips.csv", windows)):
        with open(out_dir / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    groups = [(w, lambda r, w=w: r["weather"] == w) for w in sorted({r["weather"] for r in rows})]
    groups += [("all", lambda r: True), ("suitable", lambda r: r["suitable"]),
               ("unsuitable", lambda r: not r["suitable"])]
    counting = [f"{'group':10s} {'seqs':>4s} {'MAE':>5s} {'bias':>6s} {'±1':>5s} {'±2':>5s} | "
                f"{'P50':>5s} {'R50':>5s} {'P70':>5s} {'R70':>5s} | {'IDsw/100':>8s}"]
    detail = [f"{'group':10s} | {'band P50':>8s} {'band R50':>8s} {'band IDsw/100':>13s} | recall50 by box height: "
              f"{'<25':>5s} {'25-50':>5s} {'50-100':>6s} {'>=100':>5s}"]
    crossing = [f"{'group':10s} | {'crossings':>9s} {'correct':>7s} {'missed':>6s} {'+same':>6s} {'+none':>6s} "
                f"{'+GTnot':>6s} | {'30s clips':>9s} {'avg cars':>8s} {'exact':>6s} {'±1':>5s} {'±2':>5s} {'MAE':>5s}"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = None
        for label, keep in groups:
            group = [r for r in rows if keep(r)]
            if not group:
                continue
            s = summarize(group, [w for w in windows if keep(w)])
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=["group", *s.keys()])
                writer.writeheader()
            writer.writerow({"group": label, **{k: round(v, 4) for k, v in s.items()}})
            counting.append(f"{label:10s} {s['sequences']:4d} {s['mae']:5.2f} {s['bias']:+6.2f} {s['within_1']:5.0%} "
                            f"{s['within_2']:5.0%} | {s['precision_50']:5.3f} {s['recall_50']:5.3f} "
                            f"{s['precision_70']:5.3f} {s['recall_70']:5.3f} | {s['id_switches_per_100_vehicles']:8.1f}")
            detail.append(f"{label:10s} | {s['band_precision_50']:8.3f} {s['band_recall_50']:8.3f} "
                          f"{s['band_id_switches_per_100_vehicles']:13.1f} | {'':24s}"
                          f"{s['recall_50_lt25']:5.3f} {s['recall_50_25to50']:5.3f} {s['recall_50_50to100']:6.3f} "
                          f"{s['recall_50_ge100']:5.3f}")
            crossing.append(f"{label:10s} | {s['gt_crossings']:9d} {s['crossings_correct']:7.1%} "
                            f"{s['crossings_missed']:6.1%} {s['crossings_extra_same_vehicle']:6.1%} "
                            f"{s['crossings_extra_no_vehicle']:6.1%} {s['crossings_extra_gt_not_crossing']:6.1%} | "
                            f"{s['clips']:9d} {s['clip_mean_gt_count']:8.1f} {s['clip_exact']:6.0%} "
                            f"{s['clip_within_1']:5.0%} {s['clip_within_2']:5.0%} {s['clip_mae']:5.2f}")

    print(f"\n{tag} (reference point {args.ref_point:g} of the box height)\n" + "\n".join(counting)
          + "\n\n" + "\n".join(detail) + "\n\n" + "\n".join(crossing))
    print(f"\nper-sequence: {out_dir / 'per_sequence.csv'}")
    print(f"30 s clips: {out_dir / 'clips.csv'}")
    print(f"summary: {out_dir / 'summary.csv'}")
    print(f"line images and line files: {lines_dir}")

