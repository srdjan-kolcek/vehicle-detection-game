"""RNG audit log: replaying rows, and the row open_round writes (DB tests need TEST_DATABASE_URL)."""
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from app.engine.rounds import CannotOpenRound
from app.models import City, RngAudit, Round
from app.rng.audit import audit_problems, audit_row, replay, verify_round
from app.rng.draw import RAW_BITS, draw_clip
from tests.test_engine_loop import _open, _setup_city, sessions  # noqa: F401 (sessions is a fixture)

POOL = [10, 20, 30, 40, 50]  # K = 1


def _row(**changes) -> RngAudit:
    draw = draw_clip(POOL, [30], lambda bits: 7)  # candidates [10, 20, 40, 50], 7 mod 4 = 3 -> 50
    row = audit_row(round_id=1, city_id=1, draw=draw)
    for name, value in changes.items():
        setattr(row, name, value)
    return row


# --- replay ---

def test_audit_row_copies_the_draw():
    row = _row()
    assert (row.pool_size, row.pool_ids, row.cooldown_ids, row.chosen_clip_id, row.raw_random) == (
        5, POOL, [30], 50, "7"
    )


def test_replay_matches_the_draw_for_real_csprng_values():
    pool = list(range(1, 13))
    for _ in range(500):
        draw = draw_clip(pool, [3, 7, 11])
        assert replay(draw.pool_ids, draw.cooldown_ids, draw.raw_random) == draw.clip_id


def test_replay_rejects_a_value_from_the_biased_tail():
    limit = (2**RAW_BITS // 3) * 3
    assert replay([1, 2, 3], [], limit - 1) == [1, 2, 3][(limit - 1) % 3]
    with pytest.raises(ValueError, match="accepted range"):
        replay([1, 2, 3], [], limit)


def test_a_good_row_has_no_problems():
    assert audit_problems(_row(), round_clip_id=50) == []


@pytest.mark.parametrize(
    ("changes", "round_clip_id", "problem"),
    [
        ({"raw_random": "8"}, 50, "replay gives clip 10"),
        ({"raw_random": "-1"}, 50, "cannot replay"),
        ({"raw_random": "abc"}, 50, "cannot replay"),
        ({"cooldown_ids": [50]}, 50, "replay gives clip 40"),
        ({"cooldown_ids": [30, 40]}, 50, "more than K"),
        ({"cooldown_ids": [99]}, 50, "not in the pool"),
        ({"pool_ids": [10, 20, 30, 40, 50, 60]}, 50, "pool_size 5"),
        ({"pool_ids": [50, 40, 30, 20, 10]}, 50, "not sorted"),
        ({"chosen_clip_id": 40}, 40, "replay gives clip 50"),
        ({}, 40, "round played clip 40"),
    ],
)
def test_tampered_rows_are_reported(changes, round_clip_id, problem):
    problems = audit_problems(_row(**changes), round_clip_id)
    assert any(problem in p for p in problems), problems


# --- written by open_round ---

@pytest.mark.asyncio
async def test_opening_a_round_writes_one_replayable_audit_row(session, sessions):  # noqa: F811
    city = await _setup_city(session)
    round_ = await _open(sessions, city)
    async with sessions() as s:
        rows = list(await s.scalars(select(RngAudit)))
        pool = sorted(await s.scalars(text("SELECT id FROM clips WHERE city_id = :c").bindparams(c=city.id)))
        assert await verify_round(s, round_.id) == []
    assert len(rows) == 1
    row = rows[0]
    assert (row.round_id, row.city_id, row.chosen_clip_id) == (round_.id, city.id, round_.clip_id)
    assert (row.pool_ids, row.pool_size, row.cooldown_ids) == (pool, 12, [])
    assert 0 <= int(row.raw_random) < 2**RAW_BITS


@pytest.mark.asyncio
async def test_later_rounds_record_their_cooldown(session, sessions):  # noqa: F811
    city = await _setup_city(session)  # 12 clips, K = 3
    played = []
    for _ in range(5):
        round_ = await _open(sessions, city)
        played.insert(0, round_.clip_id)
        async with sessions() as s:
            (await s.get(Round, round_.id)).status = "VOIDED"
            await s.commit()
    async with sessions() as s:
        rows = list(await s.scalars(select(RngAudit).order_by(RngAudit.id)))
        for row in rows:
            assert await verify_round(s, row.round_id) == []
    assert rows[-1].cooldown_ids == played[1:4]


@pytest.mark.asyncio
async def test_replay_survives_a_later_change_to_the_pool(session, sessions):  # noqa: F811
    city = await _setup_city(session)
    round_ = await _open(sessions, city)
    async with sessions() as s:
        await s.execute(text("UPDATE clips SET verified = false WHERE city_id = :c"), {"c": city.id})
        await s.commit()
        assert await verify_round(s, round_.id) == []


@pytest.mark.asyncio
async def test_a_failed_open_writes_no_audit_row(session, sessions):  # noqa: F811
    city = await _setup_city(session, clips=0)
    with pytest.raises(CannotOpenRound):
        await _open(sessions, city)
    async with sessions() as s:
        assert list(await s.scalars(select(RngAudit))) == []


@pytest.mark.asyncio
async def test_written_rows_cannot_be_changed(session, sessions):  # noqa: F811
    city = await _setup_city(session)
    round_ = await _open(sessions, city)
    async with sessions() as s:
        with pytest.raises(DBAPIError, match="append-only"):
            await s.execute(text("UPDATE rng_audit SET raw_random = '0' WHERE round_id = :r"), {"r": round_.id})


@pytest.mark.asyncio
async def test_a_round_without_an_audit_row_fails_verification(sessions):  # noqa: F811
    async with sessions() as s:
        assert await verify_round(s, 12345) == ["expected one audit row, found 0"]
