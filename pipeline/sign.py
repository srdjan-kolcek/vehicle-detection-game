"""Hash and sign every verified clip so the backend can refuse clips that were changed.

Per clip: sha256 of the MP4, of timelines/<clip>.json and of tracks/<clip>.json, and an
HMAC-SHA256 signature (hex, 64 characters) with PIPELINE_SIGNING_KEY over the signed record.

Signed record: exactly the clips-table columns the backend stores, plus the city slug, so the
backend can rebuild it from a row and check it:
  angle_id, city, count, duration_s, file, licence, pipeline_version, sha256, source_dataset,
  timeline_file, timeline_sha256, tracks_file, tracks_sha256, weights_sha256
Canonical form: json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
encoded as UTF-8. File paths are relative to the media root (clips/, timelines/, tracks/), as
ingest lays them out.

The key comes from the PIPELINE_SIGNING_KEY environment variable, or else from the .env file at
the project root; it is never printed. Only clips marked verified in data/clips/verify.csv are
signed. The video hash must equal the one slice recorded, and the count must equal the one in
counts.csv, so a file changed after those steps stops the run. data/clips/signed.json holds every
record with its signature, and each signature is checked again after writing.
"""
import csv
import hashlib
import hmac
import json
import os

from .detect import TIMELINES, TRACKS
from .inventory import CLIPS, DATASET, LICENCE, ROOT
from .slice import VIDEO
from .train import sha256

ALGORITHM = "HMAC-SHA256"


def load_key():
    key = os.environ.get("PIPELINE_SIGNING_KEY")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "PIPELINE_SIGNING_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("PIPELINE_SIGNING_KEY is not set (environment or .env)")
    return key


def canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def signature(record, key):
    return hmac.new(key.encode("utf-8"), canonical(record), hashlib.sha256).hexdigest()


def read_csv(name):
    with open(CLIPS / name, newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def sign(args):
    key = load_key()
    inventory, sliced, counts, verified = (read_csv(n) for n in
                                           ("inventory.csv", "slice.csv", "counts.csv", "verify.csv"))
    out = []
    for clip_id, row in inventory.items():
        if verified.get(clip_id, {}).get("verified") != "True":
            print(f"{clip_id}: not verified, not signed")
            continue
        video_sha = sha256(VIDEO / f"{clip_id}.mp4")
        if video_sha != sliced[clip_id]["sha256"]:
            raise SystemExit(f"{clip_id}: video changed since slice; run slice again")
        timeline = json.loads((TIMELINES / f"{clip_id}.json").read_text())
        if timeline["count"] != int(counts[clip_id]["count"]):
            raise SystemExit(f"{clip_id}: timeline count differs from counts.csv; run count again")
        record = {
            "angle_id": row["angle_id"],
            "city": row["city"],
            "count": timeline["count"],
            "duration_s": float(row["duration_s"]),
            "file": f"clips/{clip_id}.mp4",
            "licence": LICENCE,
            "pipeline_version": counts[clip_id]["pipeline_version"],
            "sha256": video_sha,
            "source_dataset": DATASET,
            "timeline_file": f"timelines/{clip_id}.json",
            "timeline_sha256": sha256(TIMELINES / f"{clip_id}.json"),
            "tracks_file": f"tracks/{clip_id}.json",
            "tracks_sha256": sha256(TRACKS / f"{clip_id}.json"),
            "weights_sha256": counts[clip_id]["weights_sha256"],
        }
        out.append({"clip_id": clip_id, "record": record, "signature": signature(record, key)})

    manifest = {"algorithm": ALGORITHM,
                "canonical": 'json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False), UTF-8',
                "clips": out}
    target = CLIPS / "signed.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    reread = json.loads(target.read_text())
    bad = [c["clip_id"] for c in reread["clips"]
           if not hmac.compare_digest(signature(c["record"], key), c["signature"])]
    if bad:
        raise SystemExit(f"signature check failed after writing: {', '.join(bad)}")
    per_city = {}
    for c in out:
        per_city[c["record"]["city"]] = per_city.get(c["record"]["city"], 0) + 1
    print("signed per city: " + ", ".join(f"{city} {n}" for city, n in per_city.items()))
    print(f"clips signed: {len(out)} ({ALGORITHM}); every signature re-checked")
    if key.startswith("dev-only"):
        print("note: signed with the dev-only key from .env; use a real key for anything beyond local dev")
    print(f"manifest: {target}")
