"""Round engine against a real database (needs TEST_DATABASE_URL). Time is faked, so every test runs instantly."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.engine import loop as loop_module
from app.engine.loop import CityLoop, RoundEngine, try_lock_city
from app.engine.rounds import CannotOpenRound, advance, open_round, settle_round, void_round
from app.engine.schedule import PHASE_DURATIONS_KEY, PhaseDurations
from app.media.signing import clip_record, sign_record
from app.models import AppConfig, City, Clip, PayoutTable, Round
from tests.test_rng_pool import KEY, _clip

pytestmark = pytest.mark.asyncio

START = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
LINE = 11


def _s(seconds: float) -> datetime:
    return START + timedelta(seconds=seconds)


class FakeTime:
    """A clock that only moves when the engine sleeps."""

    def __init__(self) -> None:
        self.now = START
        self.sleeps: list[float] = []

    def clock(self) -> datetime:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)
        await asyncio.sleep(0)


@pytest_asyncio.fixture()
async def sessions(async_engine):
    return async_sessionmaker(async_engine, expire_on_commit=False)


async def _setup_city(session, *, clips=12, table=True, active=True, slug="las_vegas") -> City:
    city = City(slug=slug, name_key=f"city.{slug}", lat=0, lng=0, theme_key="vegas", line_l=LINE,
                min_stake=100, max_stake=10000, active=active)
    session.add(city)
    await session.flush()
    for n in range(clips):
        clip = _clip(city_id=city.id, file=f"clips/{slug}-{n}.mp4", count=n + 5, verified=True)
        clip.signature = sign_record(clip_record(clip, slug), KEY)
        session.add(clip)
    if table:
        session.add(PayoutTable(city_id=city.id, version=1, p_under=.5, p_exact=.1, p_over=.4, m_under=1.8,
                                m_exact=8, m_over=2.25, rtp_under=.9, rtp_exact=.8, rtp_over=.9, active=True))
    await session.commit()
    return city


def _loop(sessions, city, time, events=None, **kwargs) -> CityLoop:
    async def record(event):
        if events is not None:
            events.append(event)

    return CityLoop(sessions, city.id, KEY, clock=time.clock, sleep=time.sleep, on_phase=record, **kwargs)


async def _open(sessions, city, at=START, durations=PhaseDurations()) -> Round:
    async with sessions() as s:
        round_ = await open_round(s, await s.get(City, city.id), durations, at, KEY)
        await s.commit()
    return round_


async def _set_status(sessions, round_id, status) -> None:
    async with sessions() as s:
        (await s.get(Round, round_id)).status = status
        await s.commit()


async def _round(sessions, round_id) -> Round:
    async with sessions() as s:
        return await s.get(Round, round_id)


# --- one full round ---

async def test_a_round_runs_every_phase_on_its_deadlines(session, sessions):
    city = await _setup_city(session)
    time = FakeTime()
    seen = []

    async def record(event):
        seen.append((event.phase, event.ends_at, time.now, (await _round(sessions, event.round_id)).status))

    await CityLoop(sessions, city.id, KEY, clock=time.clock, sleep=time.sleep, on_phase=record).play_round()

    assert seen == [
        ("BETTING", _s(15), _s(0), "BETTING"),
        ("LOCKED", _s(20), _s(15), "LOCKED"),
        ("PLAYBACK", _s(50), _s(20), "PLAYBACK"),
        ("SETTLE", _s(55), _s(50), "SETTLED"),
    ]
    assert time.now == _s(55)  # the settle window is waited out before returning
    async with sessions() as s:
        round_ = await s.scalar(select(Round))
    assert (round_.betting_opens_at, round_.locks_at, round_.playback_starts_at) == (_s(0), _s(15), _s(20))
    assert (round_.status, round_.settled_at) == ("SETTLED", _s(50))


async def test_result_is_the_drawn_clips_stored_count(session, sessions):
    city = await _setup_city(session)
    await _loop(sessions, city, FakeTime()).play_round()
    async with sessions() as s:
        round_ = await s.scalar(select(Round))
        clip = await s.get(Clip, round_.clip_id)
    assert round_.result_count == clip.count
    expected = "under" if clip.count < LINE else "exact" if clip.count == LINE else "over"
    assert round_.result_outcome == expected


async def test_durations_come_from_app_config(session, sessions):
    city = await _setup_city(session)
    session.add(AppConfig(key=PHASE_DURATIONS_KEY, value={"betting": 2, "locked": 1, "playback": 3, "settle": 1}))
    await session.commit()
    events = []
    await _loop(sessions, city, FakeTime(), events).play_round()
    assert [(e.phase, e.ends_at) for e in events] == [
        ("BETTING", _s(2)), ("LOCKED", _s(3)), ("PLAYBACK", _s(6)), ("SETTLE", _s(7))
    ]


async def test_bad_durations_fall_back_to_the_defaults(session, sessions):
    city = await _setup_city(session)
    session.add(AppConfig(key=PHASE_DURATIONS_KEY, value={"betting": -1}))
    await session.commit()
    events = []
    await _loop(sessions, city, FakeTime(), events).play_round()
    assert events[-1].ends_at == _s(55)


async def test_consecutive_rounds_follow_each_other_and_respect_the_cooldown(session, sessions):
    city = await _setup_city(session)  # 12 clips, K = 3
    time = FakeTime()
    loop = _loop(sessions, city, time)
    for _ in range(8):
        await loop.play_round()
    async with sessions() as s:
        rounds = list(await s.scalars(select(Round).order_by(Round.id)))
    assert [r.betting_opens_at for r in rounds] == [_s(55 * i) for i in range(8)]
    assert all(r.status == "SETTLED" for r in rounds)
    clips = [r.clip_id for r in rounds]
    for i, clip_id in enumerate(clips):
        assert clip_id not in clips[max(0, i - 3):i]


# --- cannot open ---

@pytest.mark.parametrize(
    ("setup", "reason"),
    [({"clips": 0}, "no drawable clips"), ({"table": False}, "no active payout table"),
     ({"active": False}, "inactive")],
)
async def test_city_that_cannot_play_raises(session, sessions, setup, reason):
    city = await _setup_city(session, **setup)
    with pytest.raises(CannotOpenRound, match=reason):
        await _loop(sessions, city, FakeTime()).play_round()


async def test_city_that_cannot_play_waits_and_retries(session, sessions):
    city = await _setup_city(session, clips=0)
    time = FakeTime()

    async def sleep(seconds):
        # Stop the loop at a sleep, never mid-query: cancelling psycopg inside a query can hang on Windows.
        if len(time.sleeps) == 3:
            raise asyncio.CancelledError
        await time.sleep(seconds)

    loop = CityLoop(sessions, city.id, KEY, clock=time.clock, sleep=sleep, retry_s=10)
    with pytest.raises(asyncio.CancelledError):
        await loop.run()
    assert time.sleeps == [10, 10, 10]
    async with sessions() as s:
        assert await s.scalar(select(Round)) is None


# --- guarded state changes ---

async def test_state_changes_are_guarded_and_idempotent(session, sessions):
    city = await _setup_city(session)
    round_ = await _open(sessions, city)
    async with sessions() as s:
        assert await settle_round(s, round_.id, _s(1)) is None  # not in PLAYBACK yet
        assert await advance(s, round_.id, "BETTING", "LOCKED")
        assert not await advance(s, round_.id, "BETTING", "LOCKED")  # second time does nothing
        assert not await advance(s, round_.id, "PLAYBACK", "LOCKED")
        assert await advance(s, round_.id, "LOCKED", "PLAYBACK")
        assert (await settle_round(s, round_.id, _s(50))).status == "SETTLED"
        assert await settle_round(s, round_.id, _s(51)) is None
        assert not await void_round(s, round_.id, _s(52))  # a settled round cannot be voided
        await s.commit()
    assert (await _round(sessions, round_.id)).settled_at == _s(50)


async def test_database_allows_one_open_round_per_city(session, sessions):
    city = await _setup_city(session)
    other = await _setup_city(session, slug="tokyo")
    await _open(sessions, city)
    await _open(sessions, other)  # other cities are independent
    with pytest.raises(IntegrityError, match="uq_rounds_city_id_open"):
        await _open(sessions, city)


# --- restart (mixed rule) ---

async def test_restart_resumes_betting_before_its_lock(session, sessions):
    city = await _setup_city(session)
    round_ = await _open(sessions, city)
    time, events = FakeTime(), []
    time.now = _s(5)
    await _loop(sessions, city, time, events).recover()
    assert [e.phase for e in events] == ["LOCKED", "PLAYBACK", "SETTLE"]
    assert events[0].ends_at == _s(20)
    assert (await _round(sessions, round_.id)).status == "SETTLED"


async def test_restart_voids_betting_that_missed_its_lock(session, sessions):
    city = await _setup_city(session)
    round_ = await _open(sessions, city)
    time, events = FakeTime(), []
    time.now = _s(15)
    await _loop(sessions, city, time, events).recover()
    assert [(e.phase, e.ends_at) for e in events] == [("VOIDED", None)]
    voided = await _round(sessions, round_.id)
    assert (voided.status, voided.settled_at, voided.result_count) == ("VOIDED", _s(15), None)


async def test_restart_resumes_locked_before_playback(session, sessions):
    city = await _setup_city(session)
    round_ = await _open(sessions, city)
    await _set_status(sessions, round_.id, "LOCKED")
    time, events = FakeTime(), []
    time.now = _s(18)
    await _loop(sessions, city, time, events).recover()
    assert [e.phase for e in events] == ["PLAYBACK", "SETTLE"]
    assert (await _round(sessions, round_.id)).status == "SETTLED"


async def test_restart_voids_locked_that_missed_playback(session, sessions):
    city = await _setup_city(session)
    round_ = await _open(sessions, city)
    await _set_status(sessions, round_.id, "LOCKED")
    time, events = FakeTime(), []
    time.now = _s(21)
    await _loop(sessions, city, time, events).recover()
    assert [e.phase for e in events] == ["VOIDED"]
    assert (await _round(sessions, round_.id)).status == "VOIDED"


@pytest.mark.parametrize(("restart_at", "settled_at"), [(30, 50), (90, 90)])
async def test_restart_settles_playback(session, sessions, restart_at, settled_at):
    """Mid-playback it settles on schedule; long after, it settles straight away."""
    city = await _setup_city(session)
    round_ = await _open(sessions, city)
    await _set_status(sessions, round_.id, "PLAYBACK")
    time, events = FakeTime(), []
    time.now = _s(restart_at)
    await _loop(sessions, city, time, events).recover()
    assert [e.phase for e in events] == ["SETTLE"]
    assert (await _round(sessions, round_.id)).settled_at == _s(settled_at)


async def test_restart_waits_out_the_last_settle_window(session, sessions):
    city = await _setup_city(session)
    round_ = await _open(sessions, city)
    async with sessions() as s:
        await advance(s, round_.id, "BETTING", "LOCKED")
        await advance(s, round_.id, "LOCKED", "PLAYBACK")
        await settle_round(s, round_.id, _s(50))
        await s.commit()
    time, events = FakeTime(), []
    time.now = _s(52)
    await _loop(sessions, city, time, events).recover()
    assert events == []
    assert time.now == _s(55)


# --- failures ---

async def test_failure_voids_the_round_and_the_loop_carries_on(session, sessions, monkeypatch):
    city = await _setup_city(session)
    real_settle = loop_module.settle_round
    calls = 0

    async def flaky_settle(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("database hiccup")
        return await real_settle(*args, **kwargs)

    monkeypatch.setattr(loop_module, "settle_round", flaky_settle)
    events = []

    async def record(event):
        events.append(event.phase)
        if event.phase == "SETTLE":
            raise asyncio.CancelledError  # stop here, between queries (see the retry test above)

    time = FakeTime()
    loop = CityLoop(sessions, city.id, KEY, clock=time.clock, sleep=time.sleep, on_phase=record, retry_s=10)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(loop.run(), timeout=10)

    assert events[:8] ==["BETTING", "LOCKED", "PLAYBACK", "VOIDED", "BETTING", "LOCKED", "PLAYBACK", "SETTLE"]
    async with sessions() as s:
        statuses = list(await s.scalars(select(Round.status).order_by(Round.id)))
    assert statuses[:2] == ["VOIDED", "SETTLED"]


async def test_a_failing_listener_does_not_stop_the_round(session, sessions):
    city = await _setup_city(session)

    async def broken(_):
        raise RuntimeError("websocket down")

    time = FakeTime()
    await CityLoop(sessions, city.id, KEY, clock=time.clock, sleep=time.sleep, on_phase=broken).play_round()
    async with sessions() as s:
        assert await s.scalar(select(Round.status)) == "SETTLED"


# --- one process per city ---

async def test_only_one_connection_holds_a_city_lock(async_engine):
    locks = create_async_engine(async_engine.url, poolclass=NullPool)
    try:
        async with locks.connect() as first:
            first = await first.execution_options(isolation_level="AUTOCOMMIT")
            assert await try_lock_city(first, 1)
            async with locks.connect() as second:
                second = await second.execution_options(isolation_level="AUTOCOMMIT")
                assert not await try_lock_city(second, 1)
                assert await try_lock_city(second, 2)  # other cities are free
        async with locks.connect() as third:  # closing the first connection released city 1
            third = await third.execution_options(isolation_level="AUTOCOMMIT")
            assert await try_lock_city(third, 1)
    finally:
        await locks.dispose()


async def test_engine_starts_a_round_per_active_city_and_stops(session, async_engine, sessions):
    await _setup_city(session, slug="las_vegas")
    await _setup_city(session, slug="tokyo")
    await _setup_city(session, slug="berlin", active=False)
    engine = RoundEngine(async_engine, KEY, retry_s=0.05)
    await engine.start()
    try:
        for _ in range(100):
            async with sessions() as s:
                open_rounds = list(await s.scalars(select(Round).where(Round.status == "BETTING")))
            if len(open_rounds) == 2:
                break
            await asyncio.sleep(0.05)
    finally:
        await engine.stop()
    async with sessions() as s:
        slugs = set(await s.scalars(select(City.slug).join(Round, Round.city_id == City.id)))
    assert slugs == {"las_vegas", "tokyo"}
