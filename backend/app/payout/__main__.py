"""Change the target return to player: python -m app.payout --rtp 0.80 --rtp-exact 0.75

Without --apply it only prints the new tables and changes nothing.
"""
import argparse
import asyncio
import sys

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import get_engine
from app.payout import BETS, TargetRtp
from app.payout.retarget import load_targets, retarget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m app.payout", description=__doc__.splitlines()[0])
    parser.add_argument("--rtp", type=float, help="target return to player for Under and Over, e.g. 0.90")
    parser.add_argument("--rtp-exact", type=float, help="target return to player for Exact, e.g. 0.80")
    parser.add_argument("--apply", action="store_true", help="write the targets and tables to the database")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = get_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            current = await load_targets(session)
            targets = TargetRtp(
                under_over=current.under_over if args.rtp is None else args.rtp,
                exact=current.exact if args.rtp_exact is None else args.rtp_exact,
            )
            try:
                tables = await retarget(session, targets, settings.pipeline_signing_key)
            except ValueError as exc:
                print(f"nothing changed: {exc}")
                return 1
            if args.apply:
                await session.commit()
    finally:
        await engine.dispose()

    print(f"target RTP: Under/Over {targets.under_over:.2f}, Exact {targets.exact:.2f}"
          f" (was {current.under_over:.2f} / {current.exact:.2f})\n")
    print(f"{'city':12s} {'clips':>5s} {'L':>4s} | " + " | ".join(f"{b:>15s}" for b in BETS) + " | table")
    for t in tables:
        cells = [f"{t.row[f'm_{b}']:6.2f}x {t.row[f'rtp_{b}']:6.2%}" for b in BETS]
        version = "unchanged" if t.version is None else f"new v{t.version}"
        print(f"{t.slug:12s} {t.pool_size:5d} {t.line:4d} | " + " | ".join(f"{c:>15s}" for c in cells) + f" | {version}")
    print("\neach cell: multiplier, effective RTP")
    print("written to the database" if args.apply else "dry run, nothing written; add --apply to save")
    return 0


if __name__ == "__main__":
    # psycopg's async mode cannot run on the default Windows event loop, so ask for the selector loop there.
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        sys.exit(runner.run(main(parse_args())))
