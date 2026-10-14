"""Round engine: one asyncio task per active city (doc 05, engine rules).

Each city loop opens a round, then sleeps until each fixed deadline and moves the round on:
BETTING -> LOCKED -> PLAYBACK -> SETTLED, followed by the settle window before the next round.
"""
import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.engine.rounds import (
    CannotOpenRound,
    advance,
    latest_round,
    load_durations,
    open_round,
    open_round_of,
    settle_round,
    void_round,
)
from app.engine.schedule import PhaseDurations, schedule_of
from app.models import City, Round

log = logging.getLogger(__name__)

# First key of the two-key advisory lock, so the engine's locks never clash with other advisory locks.
LOCK_NAMESPACE = 0x524E44  # "RND"
RETRY_S = 10.0


@dataclass(frozen=True)
class PhaseEvent:
    city_id: int
    round_id: int
    phase: str  # BETTING | LOCKED | PLAYBACK | SETTLE | VOIDED
    ends_at: datetime | None  # None for VOIDED


Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]
PhaseListener = Callable[[PhaseEvent], Awaitable[None]]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def _no_listener(_: PhaseEvent) -> None:
    pass


async def try_lock_city(conn: AsyncConnection, city_id: int) -> bool:
    """Session-level advisory lock: held until unlocked or the connection closes."""
    return bool(
        await conn.scalar(
            text("SELECT pg_try_advisory_lock(CAST(:ns AS integer), CAST(:city AS integer))"),
            {"ns": LOCK_NAMESPACE, "city": city_id},
        )
    )


class CityLoop:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        city_id: int,
        signing_key: str,
        *,
        clock: Clock = utc_now,
        sleep: Sleep = asyncio.sleep,
        on_phase: PhaseListener = _no_listener,
        retry_s: float = RETRY_S,
    ) -> None:
        self._sessions = sessions
        self._city_id = city_id
        self._signing_key = signing_key
        self._clock = clock
        self._sleep = sleep
        self._on_phase = on_phase
        self._retry_s = retry_s
        self._current: int | None = None  # open round this loop is driving
        self._idle_reason: str | None = None

    async def run(self) -> None:
        """Recover, then play rounds until cancelled.

        An error voids the round in progress and never stops the loop; the next pass recovers first,
        so a round that could not be voided is still closed before a new one opens.
        """
        recovering = True
        while True:
            try:
                if recovering:
                    recovering = False
                    await self.recover()
                else:
                    await self.play_round()
            except CannotOpenRound as exc:
                if str(exc) != self._idle_reason:
                    log.warning("%s; retrying every %s s", exc, self._retry_s)
                    self._idle_reason = str(exc)
                await self._sleep(self._retry_s)
            except Exception:
                log.exception("round loop of city %s failed", self._city_id)
                await self._void_current()
                recovering = True
                await self._sleep(self._retry_s)

    async def recover(self) -> None:
        """Pick up after a restart.

        An open round whose next deadline is still ahead resumes on its schedule. A BETTING or LOCKED
        round that missed its deadline is voided. A PLAYBACK round is settled: its clip is already out
        and its result was fixed by the draw. With no open round, waits out the last settle window.
        """
        async with self._sessions() as session:
            durations = await self._durations(session)
            round_ = await open_round_of(session, self._city_id)
            last = await latest_round(session, self._city_id) if round_ is None else None
        if round_ is None:
            if last is not None and last.status == "SETTLED":
                await self._sleep_until(schedule_of(last, durations).ends_at)
            return
        self._current = round_.id
        now = self._clock()
        missed = (round_.status == "BETTING" and now >= round_.locks_at) or (
            round_.status == "LOCKED" and now >= round_.playback_starts_at
        )
        if missed:
            log.warning("round %s missed its %s deadline while the engine was down; voiding", round_.id, round_.status)
            await self._void(round_.id)
            return
        log.info("resuming round %s in %s", round_.id, round_.status)
        await self._drive(round_, durations)

    async def play_round(self) -> None:
        """Open a round and drive it through every phase to the end of its settle window."""
        async with self._sessions() as session:
            durations = await self._durations(session)
            city = await session.get(City, self._city_id)
            round_ = await open_round(session, city, durations, self._clock(), self._signing_key)
            await session.commit()
        if self._idle_reason is not None:
            log.info("city %s is drawing rounds again", city.slug)
            self._idle_reason = None
        self._current = round_.id
        await self._emit(round_.id, "BETTING", round_.locks_at)
        await self._drive(round_, durations)

    async def _drive(self, round_: Round, durations: PhaseDurations) -> None:
        schedule = schedule_of(round_, durations)
        status = round_.status
        if status == "BETTING":
            await self._sleep_until(schedule.locks_at)
            if not await self._advance(round_.id, "BETTING", "LOCKED"):
                return
            status = "LOCKED"
            await self._emit(round_.id, "LOCKED", schedule.playback_starts_at)
        if status == "LOCKED":
            await self._sleep_until(schedule.playback_starts_at)
            if not await self._advance(round_.id, "LOCKED", "PLAYBACK"):
                return
            status = "PLAYBACK"
            await self._emit(round_.id, "PLAYBACK", schedule.settles_at)
        if status == "PLAYBACK":
            await self._sleep_until(schedule.settles_at)
            async with self._sessions() as session:
                settled = await settle_round(session, round_.id, self._clock())
                await session.commit()
            if settled is None:
                log.warning("round %s was closed by someone else before settling", round_.id)
                self._current = None
                return
            await self._emit(round_.id, "SETTLE", schedule.ends_at)
        self._current = None
        await self._sleep_until(schedule.ends_at)

    async def _advance(self, round_id: int, from_status: str, to_status: str) -> bool:
        async with self._sessions() as session:
            moved = await advance(session, round_id, from_status, to_status)
            await session.commit()
        if not moved:
            log.warning("round %s was no longer %s; leaving it alone", round_id, from_status)
            self._current = None
        return moved

    async def _void(self, round_id: int) -> None:
        async with self._sessions() as session:
            voided = await void_round(session, round_id, self._clock())
            await session.commit()
        self._current = None
        if voided:
            await self._emit(round_id, "VOIDED", None)

    async def _void_current(self) -> None:
        if self._current is None:
            return
        try:
            await self._void(self._current)
        except Exception:
            log.exception("could not void round %s; it is closed on the next recovery", self._current)
            self._current = None

    async def _durations(self, session: AsyncSession) -> PhaseDurations:
        try:
            return await load_durations(session)
        except ValueError:
            log.exception("bad phase_durations in app_config; using the defaults")
            return PhaseDurations()

    async def _sleep_until(self, deadline: datetime) -> None:
        delay = (deadline - self._clock()).total_seconds()
        if delay > 0:
            await self._sleep(delay)

    async def _emit(self, round_id: int, phase: str, ends_at: datetime | None) -> None:
        try:
            await self._on_phase(PhaseEvent(self._city_id, round_id, phase, ends_at))
        except Exception:
            log.exception("phase listener failed for round %s (%s)", round_id, phase)


class RoundEngine:
    """Runs one CityLoop per active city, each only while this process holds the city's advisory lock.

    The locks live on their own unpooled connections, so closing one really releases the lock and
    another backend process can take the city over.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        signing_key: str,
        *,
        on_phase: PhaseListener = _no_listener,
        retry_s: float = RETRY_S,
    ) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self._lock_engine = create_async_engine(engine.url, poolclass=NullPool)
        self._signing_key = signing_key
        self._on_phase = on_phase
        self._retry_s = retry_s
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        async with self._sessions() as session:
            cities = (await session.execute(
                select(City.id, City.slug).where(City.active.is_(True)).order_by(City.id)
            )).all()
        for city_id, slug in cities:
            self._tasks.append(asyncio.create_task(self._run_city(city_id, slug), name=f"round-loop-{slug}"))
        log.info("round engine started for %d cities", len(cities))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self._lock_engine.dispose()

    async def _run_city(self, city_id: int, slug: str) -> None:
        waiting_logged = False
        while True:
            try:
                async with self._lock_engine.connect() as conn:
                    conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
                    if await try_lock_city(conn, city_id):
                        log.info("round loop for %s started", slug)
                        waiting_logged = False
                        loop = CityLoop(
                            self._sessions, city_id, self._signing_key,
                            on_phase=self._on_phase, retry_s=self._retry_s,
                        )
                        await loop.run()
                    elif not waiting_logged:
                        log.info("round loop for %s runs in another process; standing by", slug)
                        waiting_logged = True
            except Exception:
                log.exception("round loop for %s lost its lock connection", slug)
            await asyncio.sleep(self._retry_s)
