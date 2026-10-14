"""Target RTP and payout tables (the retarget tests need TEST_DATABASE_URL)."""
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.models import AppConfig, PayoutTable
from app.payout import TargetRtp, build_table, multiplier
from app.payout.retarget import load_targets, retarget
from tests.test_engine_loop import _setup_city, sessions  # noqa: F401 (sessions is a fixture)
from tests.test_rng_pool import KEY

BELGRADE = [17, 19, 20, 22, 22, 24, 25, 25, 26, 31, 41, 57]  # demo pool, L = 24


def test_demo_pool_gets_the_same_multipliers_as_the_pipeline():
    row = build_table(BELGRADE, 24, TargetRtp())
    assert (row["m_under"], row["m_exact"], row["m_over"]) == (Decimal("2.16"), Decimal("9.6"), Decimal("1.8"))
    assert (row["rtp_under"], row["rtp_exact"], row["rtp_over"]) == (Decimal("0.9"), Decimal("0.8"), Decimal("0.9"))


@pytest.mark.parametrize("rtp", [0.90, 0.85, 0.80, 0.75])
def test_any_target_is_met_from_below(rtp):
    row = build_table(BELGRADE, 24, TargetRtp(under_over=rtp, exact=rtp - 0.05))
    for bet, target in (("under", rtp), ("over", rtp), ("exact", rtp - 0.05)):
        assert Decimal(str(target)) - Decimal("0.01") <= row[f"rtp_{bet}"] <= Decimal(str(target))


def test_multiplier_floors_without_float_error():
    assert multiplier(3, 10, 0.69) == Decimal("2.3")  # 0.69 / 0.3 is 2.2999... in floats
    assert multiplier(1, 1000, 0.9) == Decimal("50")  # capped
    assert multiplier(99, 100, 0.9) == Decimal("1.01")  # at least 1.01


@pytest.mark.parametrize(
    ("counts", "line", "problem"),
    [([], 5, "empty"), ([5, 5, 6], 5, "no clips for under"), ([1] + [5] * 98 + [9], 5, "under RTP")],  # 1 in 100: 90x, capped at 50x
)
def test_pools_that_cannot_be_priced(counts, line, problem):
    with pytest.raises(ValueError, match=problem):
        build_table(counts, line, TargetRtp())


@pytest.mark.parametrize(
    "value", [[], {"exact": 1.2}, {"under_over": 0.3}, {"under_over": True}, {"house": 0.1}]
)
def test_bad_config_values_are_refused(value):
    with pytest.raises(ValueError):
        TargetRtp.from_config(value)


def test_config_keeps_defaults_for_missing_keys():
    assert TargetRtp.from_config({"exact": 0.7}) == TargetRtp(under_over=0.90, exact=0.7)


# --- retarget (database) ---

@pytest.mark.asyncio
async def test_retarget_adds_a_new_active_table_and_stores_the_targets(session, sessions):  # noqa: F811
    city = await _setup_city(session)  # counts 5..16, L = 11: 6 under, 1 exact, 5 over
    targets = TargetRtp(under_over=0.80, exact=0.75)
    async with sessions() as s:
        [result] = await retarget(s, targets, KEY)
        await s.commit()
        tables = list(await s.scalars(select(PayoutTable).order_by(PayoutTable.version)))
        assert await load_targets(s) == targets
        assert (await s.get(AppConfig, "target_rtp")).value == {"under_over": 0.80, "exact": 0.75}
    assert (result.slug, result.version) == (city.slug, 2)
    assert [(t.version, t.active) for t in tables] == [(1, False), (2, True)]
    assert (tables[1].m_under, tables[1].m_exact, tables[1].m_over) == (Decimal("1.6"), Decimal("9"), Decimal("1.92"))

    async with sessions() as s:  # running again with the same targets changes nothing
        [again] = await retarget(s, targets, KEY)
        await s.commit()
        assert again.version is None
        assert len(list(await s.scalars(select(PayoutTable)))) == 2


@pytest.mark.asyncio
async def test_retarget_refuses_a_city_it_cannot_price(session, sessions):  # noqa: F811
    await _setup_city(session, clips=2)  # counts 5 and 6, all under L = 11
    async with sessions() as s:
        with pytest.raises(ValueError, match="las_vegas: no clips for exact"):
            await retarget(s, TargetRtp(), KEY)


@pytest.mark.asyncio
async def test_missing_targets_fall_back_to_the_defaults(sessions):  # noqa: F811
    async with sessions() as s:
        assert await load_targets(s) == TargetRtp()
