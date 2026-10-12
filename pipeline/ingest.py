"""Copy the signed clips into the media volume and write clips, lines and payout tables to Postgres.

Runs in the pipeline container, which shares the media volume with the backend and reads
DATABASE_URL and PIPELINE_SIGNING_KEY from .env:
    docker compose run --rm pipeline python -m pipeline ingest

1. Every signature in data/clips/signed.json is checked; one bad signature stops the run.
   Each city's clips in signed.json must be exactly the pool its table in tables.json was made from.
2. Files go to media/clips/, media/timelines/ and media/tracks/ (paths from the signed record).
   A file already there with the right sha256 is left alone; otherwise it is copied to a
   temporary name, checked and renamed, so the backend never sees a half-written file.
3. One database transaction:
   - cities.line_l is set from tables.json (the cities themselves come from the backend seed);
   - clips are inserted or updated by (city_id, file) with verified=true; earlier UA-DETRAC clips
     of a city that are no longer in its pool are set to verified=false, never deleted, since
     rounds may point to them;
   - per city a new payout_tables version is added and made active only when probabilities,
     multipliers or RTPs differ from the active one, so re-running changes nothing.
"""
import hmac
import json
import os
import shutil
from decimal import Decimal

import psycopg

from .inventory import CLIPS, DATASET, ROOT
from .sign import load_key, signature
from .train import sha256

MEDIA = ROOT / "media"
SOURCES = {"file": CLIPS / "video", "timeline_file": CLIPS / "timelines", "tracks_file": CLIPS / "tracks"}
HASHES = {"file": "sha256", "timeline_file": "timeline_sha256", "tracks_file": "tracks_sha256"}
BETS = ("under", "exact", "over")
P_STEP = Decimal("0.00000001")  # payout_tables p_* and rtp_* are NUMERIC(10, 8)
M_STEP = Decimal("0.01")  # m_* are NUMERIC(6, 2)


def check_inputs():
    key = load_key()
    signed = json.loads((CLIPS / "signed.json").read_text(encoding="utf-8"))["clips"]
    bad = [c["clip_id"] for c in signed if not hmac.compare_digest(signature(c["record"], key), c["signature"])]
    if bad:
        raise SystemExit(f"bad signature, nothing ingested: {', '.join(bad)}")
    tables = json.loads((CLIPS / "tables.json").read_text())
    for city, table in tables.items():
        signed_ids = sorted(c["clip_id"] for c in signed if c["record"]["city"] == city)
        if signed_ids != sorted(table["clip_ids"]):
            raise SystemExit(f"{city}: signed clips differ from the pool in tables.json; run tables and sign again")
    return signed, tables


def copy_files(signed):
    copied = kept = 0
    for clip in signed:
        record = clip["record"]
        for field, source_dir in SOURCES.items():
            target = MEDIA / record[field]
            if target.exists() and sha256(target) == record[HASHES[field]]:
                kept += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            shutil.copyfile(source_dir / target.name, tmp)
            if sha256(tmp) != record[HASHES[field]]:
                tmp.unlink()
                raise SystemExit(f"{clip['clip_id']}: {field} does not match its signed hash; run sign again")
            tmp.replace(target)
            copied += 1
    return copied, kept


def payout_row(table):
    """p, m and RTP per bet type, rounded to the database columns."""
    row = {}
    for bet in BETS:
        p = (Decimal(table[f"n_{bet}"]) / Decimal(table["pool_size"])).quantize(P_STEP)
        m = Decimal(str(table[f"m_{bet}"])).quantize(M_STEP)
        row.update({f"p_{bet}": p, f"m_{bet}": m, f"rtp_{bet}": (p * m).quantize(P_STEP)})
    return row


def write_database(signed, tables):
    url = os.environ.get("DATABASE_URL", "").replace("postgresql+psycopg://", "postgresql://")
    if not url:
        raise SystemExit("DATABASE_URL is not set; run ingest with docker compose run --rm pipeline ...")
    columns = list(payout_row(next(iter(tables.values()))))
    summary = []
    with psycopg.connect(url) as conn:  # one transaction, committed when the block ends
        city_ids = dict(conn.execute("SELECT slug, id FROM cities WHERE slug = ANY(%s)", (list(tables),)).fetchall())
        missing = sorted(set(tables) - set(city_ids))
        if missing:
            raise SystemExit(f"cities not in the database: {', '.join(missing)}; start the stack once so seed runs")

        for city, table in tables.items():
            city_id = city_ids[city]
            conn.execute("UPDATE cities SET line_l = %s WHERE id = %s", (table["line_l"], city_id))

            files = []
            for clip in (c for c in signed if c["record"]["city"] == city):
                r = clip["record"]
                files.append(r["file"])
                conn.execute(
                    """
                    INSERT INTO clips (city_id, angle_id, file, timeline_file, tracks_file, duration_s, count, sha256,
                                       timeline_sha256, tracks_sha256, signature, source_dataset, licence,
                                       weights_sha256, pipeline_version, verified)
                    VALUES (%(city_id)s, %(angle_id)s, %(file)s, %(timeline_file)s, %(tracks_file)s, %(duration_s)s,
                            %(count)s, %(sha256)s, %(timeline_sha256)s, %(tracks_sha256)s, %(signature)s,
                            %(source_dataset)s, %(licence)s, %(weights_sha256)s, %(pipeline_version)s, true)
                    ON CONFLICT (city_id, file) DO UPDATE SET
                        angle_id = EXCLUDED.angle_id, timeline_file = EXCLUDED.timeline_file,
                        tracks_file = EXCLUDED.tracks_file, duration_s = EXCLUDED.duration_s, count = EXCLUDED.count,
                        sha256 = EXCLUDED.sha256, timeline_sha256 = EXCLUDED.timeline_sha256,
                        tracks_sha256 = EXCLUDED.tracks_sha256, signature = EXCLUDED.signature,
                        source_dataset = EXCLUDED.source_dataset, licence = EXCLUDED.licence,
                        weights_sha256 = EXCLUDED.weights_sha256, pipeline_version = EXCLUDED.pipeline_version,
                        verified = true
                    """,
                    {**r, "city_id": city_id, "signature": clip["signature"]},
                )
            retired = conn.execute(
                "UPDATE clips SET verified = false WHERE city_id = %s AND source_dataset = %s AND verified "
                "AND NOT (file = ANY(%s))",
                (city_id, DATASET, files),
            ).rowcount

            new = payout_row(table)
            active = conn.execute(
                f"SELECT {', '.join(columns)} FROM payout_tables WHERE city_id = %s AND active", (city_id,)
            ).fetchone()
            if active is not None and dict(zip(columns, active)) == new:
                version = "unchanged"
            else:
                conn.execute("UPDATE payout_tables SET active = false WHERE city_id = %s AND active", (city_id,))
                number = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM payout_tables WHERE city_id = %s", (city_id,)
                ).fetchone()[0]
                conn.execute(
                    f"INSERT INTO payout_tables (city_id, version, {', '.join(columns)}, active) "
                    f"VALUES (%s, %s, {', '.join(['%s'] * len(columns))}, true)",
                    (city_id, number, *new.values()),
                )
                version = f"v{number} active"
            summary.append((city, table["line_l"], len(files), retired, version))
    return summary


def ingest(args):
    signed, tables = check_inputs()
    print(f"signatures checked: {len(signed)}")
    copied, kept = copy_files(signed)
    print(f"media files: {copied} copied, {kept} already there (in {MEDIA})")
    for city, line, clips, retired, version in write_database(signed, tables):
        print(f"{city:10s} L {line:3d}  clips {clips:3d}  retired {retired}  payout table {version}")
    print("committed")
