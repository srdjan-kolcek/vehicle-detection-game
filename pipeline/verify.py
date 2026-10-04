"""Compare every clip's count with the UA-DETRAC ground truth and record the error.

The ground truth is counted the way count-eval did it: the annotated boxes of the XML are turned
into tracks by their target id, and a vehicle counts once when its counting point (REF_POINT of
the box height) first crosses the clip's line inside the clip. Same line, same rule, so the error
is the pipeline's mistake only. The ground-truth count must equal count-eval's for the same window.

Demo rule (doc 04): every clip is kept and marked verified once its error is recorded; nothing is
rejected and no count is changed. data/clips/verify.csv has one row per clip; verify_summary.csv
and the printed report give MAE, bias and the share within +-1 and +-2 per weather, per city and
overall.
"""
import csv
import json
from collections import defaultdict

from .count_eval import TEST_XML, crossing_frames, parse_test_sequence, tracks_from_frames
from .detect import REF_POINT, TIMELINES
from .inventory import CLIPS, DEFAULT_LINES_RUN, LINES


def summary(label, rows):
    errors = [r["error"] for r in rows]
    n = len(errors)
    return {
        "group": label, "clips": n,
        "mae": round(sum(map(abs, errors)) / n, 2),
        "bias": round(sum(errors) / n, 2),
        "exact": round(sum(e == 0 for e in errors) / n, 3),
        "within_1": round(sum(abs(e) <= 1 for e in errors) / n, 3),
        "within_2": round(sum(abs(e) <= 2 for e in errors) / n, 3),
        "max_abs_error": max(map(abs, errors)),
    }


def verify(args):
    with open(CLIPS / "inventory.csv", newline="") as f:
        inventory = list(csv.DictReader(f))
    eval_gt = {}
    eval_csv = DEFAULT_LINES_RUN / "clips.csv"
    if eval_csv.exists():
        with open(eval_csv, newline="") as f:
            eval_gt = {f"{r['sequence']}-{int(r['first_frame']):05d}": int(r["gt_count"]) for r in csv.DictReader(f)}

    rows, gt_cache = [], {}
    for row in inventory:
        name = row["sequence"]
        line = json.loads((LINES / f"{row['angle_id']}.json").read_text())
        key = (name, row["angle_id"])
        if key not in gt_cache:
            _, _, targets = parse_test_sequence(TEST_XML / f"{name}.xml")
            gt_cache.clear()
            gt_cache[key] = crossing_frames(tracks_from_frames(targets, REF_POINT), line["p1"], line["p2"], 0)
        first, last = int(row["first_frame"]), int(row["last_frame"])
        gt_count = sum(first <= frame <= last for frame in gt_cache[key].values())

        expected = eval_gt.get(row["clip_id"])
        if expected is not None and expected != gt_count:
            raise SystemExit(f"{row['clip_id']}: ground truth {gt_count} but count-eval had {expected}")
        count = json.loads((TIMELINES / f"{row['clip_id']}.json").read_text())["count"]
        error = count - gt_count
        rows.append({
            "clip_id": row["clip_id"], "city": row["city"], "angle_id": row["angle_id"], "weather": row["weather"],
            "count": count, "gt_count": gt_count, "error": error, "abs_error": abs(error),
            "line_note": line.get("note", ""), "verified": True,
        })

    with open(CLIPS / "verify.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[f"weather:{r['weather']}"].append(r)
    for r in rows:
        groups[f"city:{r['city']}"].append(r)
    groups["all"] = rows
    summaries = [summary(label, group) for label, group in groups.items()]
    with open(CLIPS / "verify_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"{'group':18s} {'clips':>5s} {'MAE':>5s} {'bias':>6s} {'exact':>6s} {'±1':>5s} {'±2':>5s} {'max':>4s}")
    for s in summaries:
        print(f"{s['group']:18s} {s['clips']:5d} {s['mae']:5.2f} {s['bias']:+6.2f} {s['exact']:6.0%} "
              f"{s['within_1']:5.0%} {s['within_2']:5.0%} {s['max_abs_error']:4d}")
    worst = sorted(rows, key=lambda r: -r["abs_error"])[:5]
    print("\nlargest errors: " + ", ".join(f"{r['clip_id']} ({r['weather']}) {r['error']:+d}" for r in worst))
    checked = sum(r["clip_id"] in eval_gt for r in rows)
    print(f"ground truth equal to count-eval: {checked}/{len(rows)}; all {len(rows)} clips kept (demo rule)")
    print(f"per clip: {CLIPS / 'verify.csv'}")
    print(f"summary: {CLIPS / 'verify_summary.csv'}")
