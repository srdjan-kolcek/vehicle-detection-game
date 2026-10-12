"""Round state changes. None of these commit: the caller owns the transaction."""
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.engine.schedule import PHASE_DURATIONS_KEY, PhaseDurations, schedule_from
from app.models import AppConfig, City, Clip, PayoutTable, Round
from app.rng.draw import EmptyPoolError
from app.rng.pool import draw_for_city

OPEN_STATUSES = ("BETTING", "LOCKED", "PLAYBACK")


class CannotOpenRound(Exception):
    """The city cannot start a round right now (inactive, no drawable clips, no payout table)."""


def outcome_for(count: int, line: int) -> str:
    if count < line:
        return "under"
    if count == line:
        return "exact"
    return "over"


async def load_durations(session: AsyncSession) -> PhaseDurations:
    """Phase durations from app_config, or the defaults when the key is missing. Raises ValueError."""
    value = await session.scalar(select(AppConfig.value).where(AppConfig.key == PHASE_DURATIONS_KEY))
    return PhaseDurations() if value is None else PhaseDurations.from_config(value)


async def open_round(
    session: AsyncSession, city: City, durations: PhaseDurations, now: datetime, signing_key: str
) -> Round:
    """Draw a clip and insert a new BETTING round whose deadlines start at `now`.

    Raises CannotOpenRound when the city is inactive, has no active payout table or no drawable clip.
    """
    if not city.active:
        raise CannotOpenRound(f"city {city.slug} is inactive")
    table_id = await session.scalar(
        select(PayoutTable.id).where(PayoutTable.city_id == city.id, PayoutTable.active.is_(True))
    )
    if table_id is None:
        raise CannotOpenRound(f"city {city.slug} has no active payout table")
    try:
        draw = await draw_for_city(session, city, signing_key)
    except EmptyPoolError:
        raise CannotOpenRound(f"city {city.slug} has no drawable clips") from None
    schedule = schedule_from(now, durations)
    round_ = Round(
        city_id=city.id,
        clip_id=draw.clip_id,
        status="BETTING",
        payout_table_id=table_id,
        betting_opens_at=schedule.betting_opens_at,
        locks_at=schedule.locks_at,
        playback_starts_at=schedule.playback_starts_at,
    )
    session.add(round_)
    await session.flush()
    return round_


async def advance(session: AsyncSession, round_id: int, from_status: str, to_status: str) -> bool:
    """Move a round one step on. False when it was no longer in from_status (nothing changed)."""
    result = await session.execute(
        update(Round).where(Round.id == round_id, Round.status == from_status).values(status=to_status)
    )
    return result.rowcount == 1


async def settle_round(session: AsyncSession, round_id: int, now: datetime) -> Round | None:
    """Write the result of a round in PLAYBACK: the clip's stored, signed count against the city line.

    Returns the settled round, or None when it was not in PLAYBACK (already settled or voided).
    Bet payouts are added here by the settlement step.
    """
    round_ = await session.scalar(
        select(Round).where(Round.id == round_id).with_for_update().execution_options(populate_existing=True)
    )
    if round_ is None or round_.status != "PLAYBACK":
        return None
    count = await session.scalar(select(Clip.count).where(Clip.id == round_.clip_id))
    line = await session.scalar(select(City.line_l).where(City.id == round_.city_id))
    round_.result_count = count
    round_.result_outcome = outcome_for(count, line)
    round_.status = "SETTLED"
    round_.settled_at = now
    await session.flush()
    return round_


async def void_round(session: AsyncSession, round_id: int, now: datetime) -> bool:
    """Void a round that is still open; settled_at records when it was closed. False if it was not open.

    Stake refunds are added here by the settlement step.
    """
    result = await session.execute(
        update(Round)
        .where(Round.id == round_id, Round.status.in_(OPEN_STATUSES))
        .values(status="VOIDED", settled_at=now)
    )
    return result.rowcount == 1


async def open_round_of(session: AsyncSession, city_id: int) -> Round | None:
    """The city's unfinished round, if any (the database allows at most one)."""
    return await session.scalar(
        select(Round).where(Round.city_id == city_id, Round.status.in_(OPEN_STATUSES))
    )


async def latest_round(session: AsyncSession, city_id: int) -> Round | None:
    return await session.scalar(
        select(Round).where(Round.city_id == city_id).order_by(Round.id.desc()).limit(1)
    )
