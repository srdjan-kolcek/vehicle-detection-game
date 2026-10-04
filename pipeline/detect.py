"""Detect, track and count the clips in data/clips/inventory.csv.

detect: the fine-tuned weights run with plain ByteTrack (conf 0.5, imgsz 640) over every frame of
each whole sequence, with ignored regions greyed, exactly as count-eval did. Tracking the whole
sequence and cutting it into clips afterwards means a clip never starts with a cold tracker, and
the counts are the ones behind the measured accuracy. The tracked boxes are cached per sequence in
data/clips/raw-tracks/; a count-eval cache made with the same weights and settings is copied in
instead of running the GPU again.

count: per clip, a vehicle counts once when its counting point (horizontally centred, REF_POINT of
the box height down from the top) first crosses the angle's line in pipeline/lines/, in either
direction, and that first crossing falls inside the clip. This is the count-eval rule, so every
count must equal count-eval's predicted count for the same window; any difference stops the run.
Per clip it writes:
- data/clips/timelines/<clip>.json: count, the line (normalised 0-1) and the crossing events.
- data/clips/tracks/<clip>.json: every tracked box in the clip, per track, for the overlay.
Boxes and crossings come from the same tracks, and track ids are renumbered 1..n per clip in order
of first appearance, so the overlay and the counter always agree. data/clips/counts.csv lists the
counts with the weights sha256, pipeline version and counting settings.

tracks.json layout (version 1): coordinates are integers in 1/1000 of the frame width or height;
each track lists one [x, y, w, h] per frame from its first frame to its last (x, y = top-left),
with null where the tracker lost it for a moment. A per-frame list of box objects would be up to
1.5 MB on busy clips; this layout keeps 25 fps with no interpolation in the client.
"""
import csv
import json
import shutil
from pathlib import Path

from . import PIPELINE_VERSION
from .count_eval import TEST_XML, crossing_frames, load_or_track, parse_test_sequence, tracks_from_frames
from .inventory import CLIPS, LINES, DEFAULT_LINES_RUN, ROOT
from .train import RUNS, sha256

# the accepted setup from count-eval (data/runs/count-eval/bytetrack-conf0.5-img640)
WEIGHTS = RUNS / "finetune-20260924-160316" / "weights" / "best.pt"
TRACKER = "bytetrack.yaml"
CONF = 0.5
IMGSZ = 640
REF_POINT = 0.75
SCALE = 1000
CLASSES = ["vehicle"]  # the fine-tuned model has one class
RAW_TRACKS = CLIPS / "raw-tracks"
TIMELINES = CLIPS / "timelines"
TRACKS = CLIPS / "tracks"


class TrackArgs:
    """The settings load_or_track expects from count-eval's command line."""
    weights = WEIGHTS
    tracker = TRACKER
    conf = CONF
    imgsz = IMGSZ


def read_inventory():
    with open(CLIPS / "inventory.csv", newline="") as f:
        return list(csv.DictReader(f))


def detect(args):
    rows = read_inventory()
    sequences = list(dict.fromkeys(r["sequence"] for r in rows))
    RAW_TRACKS.mkdir(parents=True, exist_ok=True)
    model_holder = {}
    for i, name in enumerate(sequences, 1):
        cached = RAW_TRACKS / f"{name}.json"
        source = DEFAULT_LINES_RUN / "tracks" / f"{name}.json"
        if not cached.exists() and source.exists():
            shutil.copyfile(source, cached)
        _, ignored, _ = parse_test_sequence(TEST_XML / f"{name}.xml")
        before = cached.stat().st_mtime_ns if cached.exists() else None
        load_or_track(model_holder, name, ignored, TrackArgs, RAW_TRACKS)
        how = "cached" if cached.stat().st_mtime_ns == before else "tracked on the GPU"
        print(f"[{i}/{len(sequences)}] {name} {how}", flush=True)
    info = {"weights": WEIGHTS.relative_to(ROOT).as_posix(), "weights_sha256": sha256(WEIGHTS),
            "tracker": TRACKER, "conf": CONF, "imgsz": IMGSZ}
    (CLIPS / "detect.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"weights sha256: {info['weights_sha256']}")
    print(f"tracks: {RAW_TRACKS}")


def norm(v, size):
    return round(min(max(v, 0.0), size) / size * SCALE)


def clip_outputs(row, frames, line, width, height):
    """Return (timeline, tracks) for one clip from the whole sequence's tracked boxes."""
    first, last = int(row["first_frame"]), int(row["last_frame"])
    fps = int(row["fps"])

    crossings = crossing_frames(tracks_from_frames(frames, REF_POINT), line["p1"], line["p2"], 0)
    inside = sorted((frame, tid) for tid, frame in crossings.items() if first <= frame <= last)

    per_track = {}  # sequence track id -> {frame number: box}
    for number in range(first, last + 1):
        for tid, x1, y1, x2, y2 in frames.get(number, []):
            per_track.setdefault(tid, {})[number] = (x1, y1, x2, y2)
    order = sorted(per_track, key=lambda tid: (min(per_track[tid]), tid))
    new_id = {tid: k for k, tid in enumerate(order, 1)}

    tracks = []
    for tid in order:
        boxes = per_track[tid]
        start, end = min(boxes), max(boxes)
        out = []
        for number in range(start, end + 1):
            if number not in boxes:
                out.append(None)
                continue
            x1, y1, x2, y2 = boxes[number]
            x, y = norm(x1, width), norm(y1, height)
            out.append([x, y, norm(x2, width) - x, norm(y2, height) - y])
        tracks.append({"id": new_id[tid], "class": 0, "start": start - first, "boxes": out})

    timeline = {
        "version": 1, "clip_id": row["clip_id"], "fps": fps, "frames": last - first + 1,
        "count": len(inside),
        "line": {"p1": [round(line["p1"][0] / width, 4), round(line["p1"][1] / height, 4)],
                 "p2": [round(line["p2"][0] / width, 4), round(line["p2"][1] / height, 4)],
                 "directions": line["directions"]},
        "events": [{"t": round((frame - first) / fps, 2), "frame": frame - first, "track_id": new_id[tid],
                    "class": CLASSES[0]} for frame, tid in inside],
    }
    track_doc = {"version": 1, "clip_id": row["clip_id"], "fps": fps, "frames": last - first + 1,
                 "scale": SCALE, "classes": CLASSES, "tracks": tracks}
    return timeline, track_doc


def write_json(path, data):
    path.write_text(json.dumps(data, separators=(",", ":")))
    return path.stat().st_size


def count(args):
    info = json.loads((CLIPS / "detect.json").read_text())
    if info["weights_sha256"] != sha256(WEIGHTS):
        raise SystemExit("weights changed since detect; run python -m pipeline detect again")
    eval_counts = {}
    eval_csv = DEFAULT_LINES_RUN / "clips.csv"
    if eval_csv.exists():
        with open(eval_csv, newline="") as f:
            eval_counts = {f"{r['sequence']}-{int(r['first_frame']):05d}": int(r["pred_count"]) for r in csv.DictReader(f)}

    TIMELINES.mkdir(parents=True, exist_ok=True)
    TRACKS.mkdir(parents=True, exist_ok=True)
    out, loaded = [], {}
    for row in read_inventory():
        name = row["sequence"]
        if name not in loaded:
            data = json.loads((RAW_TRACKS / f"{name}.json").read_text())
            loaded.clear()
            loaded[name] = {int(k): [tuple(b) for b in v] for k, v in data["frames"].items()}
        line = json.loads((LINES / f"{row['angle_id']}.json").read_text())
        width, height = line["frame_size"]
        timeline, track_doc = clip_outputs(row, loaded[name], line, width, height)

        expected = eval_counts.get(row["clip_id"])
        if expected is not None and expected != timeline["count"]:
            raise SystemExit(f"{row['clip_id']}: count {timeline['count']} but count-eval counted {expected}")
        timeline_bytes = write_json(TIMELINES / f"{row['clip_id']}.json", timeline)
        tracks_bytes = write_json(TRACKS / f"{row['clip_id']}.json", track_doc)
        out.append({
            "clip_id": row["clip_id"], "city": row["city"], "angle_id": row["angle_id"], "weather": row["weather"],
            "count": timeline["count"], "tracks": len(track_doc["tracks"]),
            "timeline_bytes": timeline_bytes, "tracks_bytes": tracks_bytes,
            "weights_sha256": info["weights_sha256"], "pipeline_version": PIPELINE_VERSION,
            "tracker": TRACKER, "conf": CONF, "imgsz": IMGSZ, "ref_point": REF_POINT,
        })
        print(f"{row['clip_id']} ({row['city']}, {row['weather']}) count {timeline['count']:3d}  "
              f"tracks {len(track_doc['tracks']):3d}  tracks.json {tracks_bytes / 1000:4.0f} KB", flush=True)

    with open(CLIPS / "counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    sizes = sorted(r["tracks_bytes"] for r in out)
    checked = sum(r["clip_id"] in eval_counts for r in out)
    print(f"\nclips: {len(out)}, counts equal to count-eval: {checked}/{len(out)}")
    print(f"tracks.json size: median {sizes[len(sizes) // 2] / 1000:.0f} KB, largest {sizes[-1] / 1000:.0f} KB")
    print(f"counts: {CLIPS / 'counts.csv'}")
