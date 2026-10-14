"""RNG audit log (doc 03, RNG).

Every draw writes one append-only rng_audit row in the same transaction as its round. The row holds
everything needed to replay the draw without trusting the current clips table: the whole pool, the
cooldown window and the accepted raw value. Replaying is raw mod n over the pool minus the cooldown,
sorted by clip id.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Round, RngAudit
from app.rng.draw import RAW_BITS, Draw, cooldown_size


def audit_row(round_id: int, city_id: int, draw: Draw) -> RngAudit:
    return RngAudit(
        round_id=round_id,
        city_id=city_id,
        pool_size=draw.pool_size,
        pool_ids=list(draw.pool_ids),
        cooldown_ids=list(draw.cooldown_ids),
        chosen_clip_id=draw.clip_id,
        raw_random=str(draw.raw_random),
    )


def replay(pool_ids: list[int], cooldown_ids: list[int], raw: int) -> int:
    """The clip a raw value picks. Raises ValueError when the draw would have rejected the value."""
    candidates = sorted(set(pool_ids) - set(cooldown_ids))
    if not candidates:
        raise ValueError("no candidates")
    n = len(candidates)
    if not 0 <= raw < (2**RAW_BITS // n) * n:
        raise ValueError("raw value is outside the accepted range")
    return candidates[raw % n]


def audit_problems(row: RngAudit, round_clip_id: int | None) -> list[str]:
    """Everything wrong with an audit row; empty when the draw replays to the round's clip."""
    problems = []
    pool, cooldown = list(row.pool_ids), list(row.cooldown_ids)
    if pool != sorted(set(pool)):
        problems.append("pool_ids are not sorted and distinct")
    if row.pool_size != len(pool):
        problems.append(f"pool_size {row.pool_size} does not match {len(pool)} pool ids")
    if len(set(cooldown)) != len(cooldown) or not set(cooldown) <= set(pool):
        problems.append("cooldown_ids repeat or are not in the pool")
    if len(cooldown) > cooldown_size(len(pool)):
        problems.append(f"cooldown has {len(cooldown)} clips, more than K = {cooldown_size(len(pool))}")
    try:
        replayed = replay(pool, cooldown, int(row.raw_random))
    except ValueError as exc:
        problems.append(f"cannot replay: {exc}")
    else:
        if replayed != row.chosen_clip_id:
            problems.append(f"replay gives clip {replayed}, the row says {row.chosen_clip_id}")
    if row.chosen_clip_id != round_clip_id:
        problems.append(f"round played clip {round_clip_id}, the row says {row.chosen_clip_id}")
    return problems


async def verify_round(session: AsyncSession, round_id: int) -> list[str]:
    """Replay a round's draw from its audit row. Empty when everything checks out."""
    rows = list(await session.scalars(select(RngAudit).where(RngAudit.round_id == round_id)))
    if len(rows) != 1:
        return [f"expected one audit row, found {len(rows)}"]
    round_clip_id = await session.scalar(select(Round.clip_id).where(Round.id == round_id))
    return audit_problems(rows[0], round_clip_id)
