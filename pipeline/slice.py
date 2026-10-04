"""Encode every clip in data/clips/inventory.csv from the UA-DETRAC JPG frames to an MP4.

Output: data/clips/video/<clip id>.mp4, H.264 (libx264, yuv420p, CRF 23 capped at 4 Mbit/s) at 25 fps
and 960x540, no audio,
moov atom at the front (+faststart) so the browser can start playing before the whole file is in.
The frames are the original ones: ignored regions are greyed only for detection, never in the video.

ffmpeg comes from the imageio-ffmpeg wheel (pinned), so no system install is needed. Metadata is
stripped and the bitexact flags are set, so the same frames and settings give the same bytes and
the same sha256. Each file is checked after encoding by reading it back with OpenCV (frame count
and size). data/clips/slice.csv lists every clip with its file, size, sha256 and the settings.

Not done for UA-DETRAC (doc 04): perceptual-hash dedupe and scene-cut/shake checks. Every clip is
a separate non-overlapping window of a fixed camera, so they cannot find anything.
"""
import csv
import hashlib
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

from .count_eval import IMAGES
from .inventory import CLIPS

VIDEO = CLIPS / "video"
WIDTH, HEIGHT = 960, 540
CRF = 23
PRESET = "slow"
MAXRATE = "4M"  # caps the few very busy scenes (MVI_39051 was 8 Mbit/s uncapped) so clips stream on phones
ENCODER_ARGS = [
    "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF), "-maxrate", MAXRATE, "-bufsize", "8M",
    "-pix_fmt", "yuv420p",
    "-vf", f"scale={WIDTH}:{HEIGHT}", "-an", "-movflags", "+faststart",
    "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(row, target):
    """Encode one clip to target, writing a temporary file first so a crash never leaves a half clip."""
    tmp = target.with_suffix(".tmp.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", row["fps"], "-start_number", row["first_frame"],
        "-i", str(IMAGES / row["sequence"] / "img%05d.jpg"),
        "-frames:v", row["frames"], *ENCODER_ARGS, str(tmp),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{row['clip_id']}: ffmpeg failed:\n{result.stderr}")
    tmp.replace(target)


def check(row, path):
    """Read the MP4 back; return an error text, or None if frame count and size are right."""
    video = cv2.VideoCapture(str(path))
    frames = 0
    size = None
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frames += 1
        size = frame.shape[1], frame.shape[0]
    video.release()
    if frames != int(row["frames"]):
        return f"{frames} frames, expected {row['frames']}"
    if size != (WIDTH, HEIGHT):
        return f"size {size}, expected {WIDTH}x{HEIGHT}"
    return None


def slice_clips(args):
    with open(CLIPS / "inventory.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    VIDEO.mkdir(parents=True, exist_ok=True)
    settings = f"libx264 preset={PRESET} crf={CRF} maxrate={MAXRATE} yuv420p {WIDTH}x{HEIGHT} faststart no-audio bitexact"
    ffmpeg_version = imageio_ffmpeg.get_ffmpeg_version()

    out, encoded = [], 0
    for i, row in enumerate(rows, 1):
        target = VIDEO / f"{row['clip_id']}.mp4"
        if args.force or not target.exists():
            encode(row, target)
            encoded += 1
        error = check(row, target)
        if error:
            raise SystemExit(f"{row['clip_id']}: bad video, {error}; delete {target} and run again")
        out.append({
            "clip_id": row["clip_id"], "city": row["city"], "file": target.name,
            "bytes": target.stat().st_size, "sha256": sha256(target),
            "duration_s": row["duration_s"], "ffmpeg": ffmpeg_version, "settings": settings,
        })
        print(f"[{i}/{len(rows)}] {row['clip_id']} {out[-1]['bytes'] / 1e6:5.1f} MB", flush=True)

    with open(CLIPS / "slice.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    total = sum(r["bytes"] for r in out) / 1e6
    print(f"\nclips: {len(out)} ({encoded} encoded, {len(out) - encoded} already there), {total:.0f} MB")
    print(f"videos: {VIDEO}")
    print(f"list: {CLIPS / 'slice.csv'}")
