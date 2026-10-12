"""Per-city line L, Under/Exact/Over probabilities and payout multipliers (doc 03).

The pool of a city is all its verified clips (data/clips/verify.csv; in the demo every clip is
kept). The clip is drawn uniformly, so the probabilities are exact frequencies of the pool:
p_under = #(count < L) / n, p_exact = #(count == L) / n, p_over = #(count > L) / n.

L: only counts that occur in the pool are candidates, so Exact can always win. Among them the one
with the smallest |p_under - p_over| wins; ties go to the larger p_exact, then the smaller L. With
one clip per count (typical for ~12 clips) this is the median clip's count.

Multiplier per bet type: floor to 0.01 of r / p, at least 1.01 and at most 50, where r is the
target RTP of that bet type: 0.90 for Under and Over (house margin 10%, usual for a main market)
and 0.80 for Exact (house margin 20%, usual for a long-shot market such as a correct score). The
house therefore keeps a margin on every bet type. The effective RTP p * m is stored, and every bet
type must land in [r - 1%, r]; otherwise the run stops. It also stops if L leaves Under or Over
with no clips (a skewed pool such as 3 3 3 3 10), since that bet could never win. data/clips/tables.json holds the tables, the sorted counts and a hash of
the pool (clip ids and counts) so ingest can tell whether a city's table changed.
"""
import csv
import hashlib
import json
import math
from collections import Counter

from .inventory import CLIPS

MIN_MULTIPLIER = 1.01
MAX_MULTIPLIER = 50.0


def choose_line(counts):
    n = len(counts)
    best = None
    for line in sorted(set(counts)):
        under = sum(c < line for c in counts)
        over = sum(c > line for c in counts)
        exact = n - under - over
        key = (abs(under - over), -exact, line)
        if best is None or key < best[0]:
            best = (key, line)
    return best[1]


def multiplier(p, rtp):
    m = math.floor(round(rtp / p * 100, 6)) / 100  # round first so 2.2999999 floors to 2.29, not 2.28
    return min(max(m, MIN_MULTIPLIER), MAX_MULTIPLIER)


def city_table(clips, targets):
    counts = sorted(c["count"] for c in clips)
    n = len(counts)
    line = choose_line(counts)
    sizes = {"under": sum(c < line for c in counts), "exact": counts.count(line), "over": sum(c > line for c in counts)}
    table = {"pool_size": n, "line_l": line}
    for bet, k in sizes.items():
        rtp = targets[bet]
        table[f"target_rtp_{bet}"] = rtp
        if k == 0:  # line sits at the edge of the pool, so this bet can never win and has no fair multiplier
            raise SystemExit(f"{clips[0]['city']}: no clips for {bet} at line L={line} (counts {counts}); "
                             f"the pool is too skewed, add clips on both sides of L")
        p = k / n
        m = multiplier(p, rtp)
        effective = round(p * m, 4)
        if not rtp - 0.01 <= effective <= rtp:
            raise SystemExit(f"{clips[0]['city']}: {bet} RTP {effective} is outside [{rtp - 0.01:.2f}, {rtp:.2f}]")
        table.update({f"n_{bet}": k, f"p_{bet}": round(p, 6), f"m_{bet}": m, f"rtp_{bet}": effective})
    table["counts"] = counts
    table["count_histogram"] = dict(sorted(Counter(counts).items()))
    pool = sorted((c["clip_id"], c["count"]) for c in clips)
    table["pool_sha256"] = hashlib.sha256(json.dumps(pool).encode()).hexdigest()
    table["clip_ids"] = [clip_id for clip_id, _ in pool]
    return table


def tables(args):
    with open(CLIPS / "verify.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["verified"] == "True"]
    by_city = {}
    for r in rows:
        by_city.setdefault(r["city"], []).append({"clip_id": r["clip_id"], "city": r["city"], "count": int(r["count"])})

    targets = {"under": args.rtp, "exact": args.rtp_exact, "over": args.rtp}
    out = {city: city_table(clips, targets) for city, clips in by_city.items()}
    (CLIPS / "tables.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"target RTP: Under/Over {args.rtp:.2f}, Exact {args.rtp_exact:.2f}\n")
    print(f"{'city':10s} {'clips':>5s} {'L':>3s} | {'under':>15s} | {'exact':>15s} | {'over':>15s} | counts")
    for city, t in out.items():
        cells = [f"{t[f'n_{b}']:2d} {t[f'p_{b}']:4.0%} {t[f'm_{b}']:5.2f}x" for b in ("under", "exact", "over")]
        print(f"{city:10s} {t['pool_size']:5d} {t['line_l']:3d} | " + " | ".join(f"{c:>15s}" for c in cells)
              + f" | {' '.join(map(str, t['counts']))}")
    print("\neach cell: clips, probability, multiplier")
    for bets in (("under", "over"), ("exact",)):
        rtps = [t[f"rtp_{b}"] for t in out.values() for b in bets]
        print(f"effective RTP {'/'.join(bets)}: {min(rtps):.4f} to {max(rtps):.4f}")
    print(f"tables: {CLIPS / 'tables.json'}")
