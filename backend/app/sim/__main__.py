"""RTP simulation of every active city: python -m app.sim --rounds 10000000 --seed 1 --out rtp_report.json

Reads the live setup from the database (drawable pool, line, active payout table, min stake, target
RTP), simulates the cities in parallel and exits with 1 when any city fails the gate.
"""
import argparse
import asyncio
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import get_engine
from app.models import City, PayoutTable
from app.payout import BETS, TargetRtp
from app.payout.retarget import load_targets
from app.rng.pool import load_pool
from app.sim.rtp import STRATEGIES, CitySetup, CityReport, gate, simulate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.sim", description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=10_000_000, help="rounds per city (default 10 million)")
    parser.add_argument("--seed", default="1", help="seed of the simulated draws; same seed, same result")
    parser.add_argument("--tolerance", type=float, default=0.01, help="allowed distance from the target RTP")
    parser.add_argument("--session", type=int, default=1000, help="rounds per player session for the lucky-player figure")
    parser.add_argument("--stake", type=int, help="stake in minor units (default: each city's minimum stake)")
    parser.add_argument("--out", type=Path, help="write the full report as JSON here")
    return parser.parse_args()


async def load_setups(stake: int | None) -> tuple[list[CitySetup], TargetRtp, list[str]]:
    settings = get_settings()
    engine = get_engine()
    setups, skipped = [], []
    try:
        async with AsyncSession(engine) as session:
            targets = await load_targets(session)
            for city in await session.scalars(select(City).where(City.active.is_(True)).order_by(City.id)):
                pool = await load_pool(session, city, settings.pipeline_signing_key)
                table = await session.scalar(
                    select(PayoutTable).where(PayoutTable.city_id == city.id, PayoutTable.active.is_(True))
                )
                if not pool or table is None:
                    skipped.append(city.slug)
                    continue
                setups.append(CitySetup(
                    slug=city.slug,
                    clip_ids=[c.id for c in pool],
                    counts=[c.count for c in pool],
                    line=city.line_l,
                    multipliers={b: getattr(table, f"m_{b}") for b in BETS},
                    stake=stake or city.min_stake,
                ))
    finally:
        await engine.dispose()
    return setups, targets, skipped


def print_report(report: CityReport, problems: list[str]) -> None:
    print(f"\n{report.slug}  ({report.rounds:,} rounds, seed {report.seed})  {'FAIL' if problems else 'pass'}")
    print(f"  {'strategy':9s} {'exact RTP':>9s} {'measured':>9s} {'bet SD':>7s} {'lucky p99':>9s}")
    for s in STRATEGIES:
        st = report.strategies[s]
        exact = f"{report.exact_rtp[s]:9.2%}" if s in report.exact_rtp else f"{'':9s}"
        print(f"  {s:9s} {exact} {st.rtp:9.2%} {st.bet_sd:7.2f} {st.session_rtp_p99:9.2%}")
    print(f"  best single round {report.best_round_rtp:.2%}, clip draw share off uniform by at most "
          f"{report.clip_share_deviation:.2%}")
    for p in problems:
        print(f"  ! {p}")


def main() -> int:
    args = parse_args()
    # psycopg's async mode cannot run on the default Windows event loop, so ask for the selector loop there.
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        setups, targets, skipped = runner.run(load_setups(args.stake))
    if not setups:
        print("no active city has a drawable pool and an active payout table")
        return 1
    print(f"target RTP: Under/Over {targets.under_over:.0%}, Exact {targets.exact:.0%}, tolerance {args.tolerance:.1%}")
    if skipped:
        print(f"skipped (no pool or payout table): {', '.join(skipped)}")

    with ProcessPoolExecutor() as pool:
        reports = list(pool.map(simulate, setups, [args.rounds] * len(setups),
                                [f"{args.seed}:{s.slug}" for s in setups], [args.session] * len(setups)))
    results = {r.slug: (r, gate(r, targets, args.tolerance)) for r in reports}
    for report, problems in results.values():
        print_report(report, problems)

    overall = {b: sum(r.strategies[b].rtp for r, _ in results.values()) / len(results) for b in BETS}
    print("\nall cities, fixed bets: " + ", ".join(f"{b} {v:.2%}" for b, v in overall.items()))
    print("lucky p99 = what the luckiest 1% of players get back over a session of "
          f"{args.session} rounds; bet SD = spread of one bet's return per credit")
    failed = [slug for slug, (_, problems) in results.items() if problems]
    print(f"gate: {'FAIL (' + ', '.join(failed) + ')' if failed else 'pass'}")

    if args.out:
        args.out.write_text(json.dumps({
            "targets": targets.as_config(),
            "tolerance": args.tolerance,
            "session_rounds": args.session,
            "cities": {slug: {**asdict(r), "problems": p} for slug, (r, p) in results.items()},
            "passed": not failed,
        }, indent=2) + "\n")
        print(f"report: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
