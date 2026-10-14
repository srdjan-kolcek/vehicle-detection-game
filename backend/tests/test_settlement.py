"""Settlement, void and rollback (needs TEST_DATABASE_URL, except the payout rounding test)."""
import asyncio
from datetime import timedelta
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.engine.loop import CityLoop
from app.engine.rounds import advance, settle_round, void_round
from app.engine.settlement import payout_for
from app.models import Bet, Clip, LedgerEntry, Round
from app.wallet.virtual import VirtualWallet
from tests.test_bets import BALANCE, _balance, _bet, _player
from tests.test_engine_loop import START, FakeTime, _loop, _open, _setup_city
from tests.test_rng_pool import KEY

pytestmark = pytest.mark.asyncio

# The test city has line 11, clips counting 5..16 and multipliers under 1.80, exact 8.00, over 2.25.
COUNT_FOR = {"under": 8, "exact": 11, "over": 14}
WIN = {"under": 1800, "exact": 8000, "over": 2250}  # payout of a 1000 stake


@pytest_asyncio.fixture()
async def sessions(async_engine):
    return async_sessionmaker(async_engine, expire_on_commit=False)


@pytest_asyncio.fixture()
async def city(session):
    return await _setup_city(session)


async def _round_with_count(sessions, city, count, at=START) -> Round:
    """Open a round and swap its drawn clip for the one with the given count."""
    round_ = await _open(sessions, city, at=at)
    async with sessions() as s:
        clip_id = await s.scalar(select(Clip.id).where(Clip.city_id == city.id, Clip.count == count))
        (await s.get(Round, round_.id)).clip_id = clip_id
        await s.commit()
    return round_


async def _to_playback(sessions, round_id) -> None:
    async with sessions() as s:
        await advance(s, round_id, "BETTING", "LOCKED")
        await advance(s, round_id, "LOCKED", "PLAYBACK")
        await s.commit()


async def _settle(sessions, round_id, at=50):
    async with sessions() as s:
        settled = await settle_round(s, round_id, START + timedelta(seconds=at))
        await s.commit()
    return settled


async def _void(sessions, round_id, at=20) -> bool:
    async with sessions() as s:
        voided = await void_round(s, round_id, START + timedelta(seconds=at))
        await s.commit()
    return voided


async def _bet_row(sessions, bet_id) -> Bet:
    async with sessions() as s:
        return await s.get(Bet, bet_id)


async def _entries(sessions, bet_id) -> list[tuple[str, int]]:
    async with sessions() as s:
        rows = await s.execute(
            select(LedgerEntry.type, LedgerEntry.amount).where(LedgerEntry.bet_id == bet_id).order_by(LedgerEntry.id)
        )
    return [tuple(r) for r in rows]


async def _three_bets(sessions, round_id) -> dict[str, tuple]:
    """One player on each choice, 1000 each. Returns {choice: (player_id, bet_id)}."""
    placed = {}
    for choice in ("under", "exact", "over"):
        player = await _player(sessions)
        bet, _ = await _bet(sessions, player, round_id, choice=choice, stake=1000)
        placed[choice] = (player, bet.id)
    return placed


# --- payout rounding ---

@pytest.mark.parametrize(
    ("stake", "multiplier", "payout"),
    [(1000, "1.80", 1800), (333, "2.04", 679), (101, "6.66", 672), (100, "1.01", 101), (1, "50.00", 50)],
)
async def test_payout_is_stake_times_multiplier_rounded_down(stake, multiplier, payout):
    assert payout_for(stake, Decimal(multiplier)) == payout


# --- settling each outcome ---

@pytest.mark.parametrize("outcome", ["under", "exact", "over"])
async def test_settlement_pays_only_the_winning_choice(sessions, city, outcome):
    round_ = await _round_with_count(sessions, city, COUNT_FOR[outcome])
    placed = await _three_bets(sessions, round_.id)
    await _to_playback(sessions, round_.id)

    settled = await _settle(sessions, round_.id)
    assert (settled.result_count, settled.result_outcome) == (COUNT_FOR[outcome], outcome)

    for choice, (player, bet_id) in placed.items():
        bet = await _bet_row(sessions, bet_id)
        if choice == outcome:
            assert (bet.status, bet.payout) == ("won", WIN[choice])
            assert await _balance(sessions, player) == BALANCE - 1000 + WIN[choice]
            assert await _entries(sessions, bet_id) == [("bet", -1000), ("payout", WIN[choice])]
        else:
            assert (bet.status, bet.payout) == ("lost", 0)
            assert await _balance(sessions, player) == BALANCE - 1000
            assert await _entries(sessions, bet_id) == [("bet", -1000), ("payout", 0)]


async def test_a_round_without_bets_settles(sessions, city):
    round_ = await _round_with_count(sessions, city, 8)
    await _to_playback(sessions, round_.id)
    assert (await _settle(sessions, round_.id)).status == "SETTLED"


# --- settling twice ---

async def test_settling_again_changes_nothing(sessions, city):
    round_ = await _round_with_count(sessions, city, COUNT_FOR["over"])
    placed = await _three_bets(sessions, round_.id)
    await _to_playback(sessions, round_.id)
    await _settle(sessions, round_.id)

    assert await _settle(sessions, round_.id, at=51) is None
    assert not await _void(sessions, round_.id, at=52)  # a settled round cannot be voided either
    player, bet_id = placed["over"]
    assert await _balance(sessions, player) == BALANCE - 1000 + WIN["over"]
    assert await _entries(sessions, bet_id) == [("bet", -1000), ("payout", WIN["over"])]


async def test_parallel_settlement_pays_once(sessions, city):
    round_ = await _round_with_count(sessions, city, COUNT_FOR["exact"])
    placed = await _three_bets(sessions, round_.id)
    await _to_playback(sessions, round_.id)

    results = await asyncio.gather(*(_settle(sessions, round_.id) for _ in range(3)))
    assert sum(r is not None for r in results) == 1
    async with sessions() as s:
        payouts = await s.scalar(select(func.count()).select_from(LedgerEntry).where(LedgerEntry.type == "payout"))
    assert payouts == 3
    assert await _balance(sessions, placed["exact"][0]) == BALANCE - 1000 + WIN["exact"]


async def test_settle_and_void_racing_do_only_one(sessions, city):
    round_ = await _round_with_count(sessions, city, COUNT_FOR["under"])
    placed = await _three_bets(sessions, round_.id)
    await _to_playback(sessions, round_.id)

    settled, voided = await asyncio.gather(_settle(sessions, round_.id), _void(sessions, round_.id))
    assert (settled is None) == voided
    async with sessions() as s:
        statuses = set(await s.scalars(select(Bet.status).where(Bet.round_id == round_.id)))
    assert statuses == ({"refunded"} if voided else {"won", "lost"})
    for _, bet_id in placed.values():
        assert len(await _entries(sessions, bet_id)) == 2  # the stake and exactly one of payout/refund


async def test_two_cities_settle_at_once_with_the_same_players(sessions, session):
    vegas, tokyo = await _setup_city(session), await _setup_city(session, slug="tokyo")
    rounds = [await _round_with_count(sessions, c, COUNT_FOR["under"]) for c in (vegas, tokyo)]
    players = [await _player(sessions) for _ in range(5)]
    for round_ in rounds:
        for player in players:
            await _bet(sessions, player, round_.id, choice="under", stake=1000)
        await _to_playback(sessions, round_.id)

    await asyncio.wait_for(asyncio.gather(*(_settle(sessions, r.id) for r in rounds)), timeout=10)
    for player in players:
        assert await _balance(sessions, player) == BALANCE + 2 * (WIN["under"] - 1000)


# --- void and rollback ---

@pytest.mark.parametrize("status", ["BETTING", "LOCKED", "PLAYBACK"])
async def test_void_refunds_every_stake(sessions, city, status):
    round_ = await _round_with_count(sessions, city, 8)
    placed = await _three_bets(sessions, round_.id)
    async with sessions() as s:
        if status != "BETTING":
            await advance(s, round_.id, "BETTING", "LOCKED")
        if status == "PLAYBACK":
            await advance(s, round_.id, "LOCKED", "PLAYBACK")
        await s.commit()

    assert await _void(sessions, round_.id)
    for player, bet_id in placed.values():
        bet = await _bet_row(sessions, bet_id)
        assert (bet.status, bet.payout) == ("refunded", 0)
        assert await _balance(sessions, player) == BALANCE
        assert await _entries(sessions, bet_id) == [("bet", -1000), ("refund", 1000)]


async def test_voiding_again_refunds_nothing_more(sessions, city):
    round_ = await _round_with_count(sessions, city, 8)
    placed = await _three_bets(sessions, round_.id)
    assert await _void(sessions, round_.id)
    assert not await _void(sessions, round_.id, at=21)
    assert await _settle(sessions, round_.id) is None
    for player, bet_id in placed.values():
        assert await _balance(sessions, player) == BALANCE
        assert len(await _entries(sessions, bet_id)) == 2


async def test_a_failed_settlement_keeps_nothing(sessions, city, monkeypatch):
    """Settlement is one transaction: a wallet error part way through leaves the round in PLAYBACK."""
    round_ = await _round_with_count(sessions, city, COUNT_FOR["over"])
    placed = await _three_bets(sessions, round_.id)
    await _to_playback(sessions, round_.id)
    real_settle = VirtualWallet.settle
    calls = 0

    async def flaky(self, bet_id, payout):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("wallet down")
        await real_settle(self, bet_id, payout)

    monkeypatch.setattr(VirtualWallet, "settle", flaky)
    with pytest.raises(RuntimeError):
        await _settle(sessions, round_.id)
    monkeypatch.setattr(VirtualWallet, "settle", real_settle)

    async with sessions() as s:
        assert (await s.get(Round, round_.id)).status == "PLAYBACK"
        assert set(await s.scalars(select(Bet.status).where(Bet.round_id == round_.id))) == {"open"}
    for _, bet_id in placed.values():
        assert await _entries(sessions, bet_id) == [("bet", -1000)]
    assert await _settle(sessions, round_.id) is not None  # it can still be settled afterwards


# --- through the engine ---

async def test_engine_round_pays_bets_placed_while_betting(sessions, city):
    time = FakeTime()
    placed = {}

    async def bet_when_open(event):
        if event.phase == "BETTING":
            async with sessions() as s:
                count = await s.scalar(select(Clip.count).join(Round, Round.clip_id == Clip.id)
                                       .where(Round.id == event.round_id))
            placed["count"] = count
            placed["bets"] = await _three_bets(sessions, event.round_id)

    await CityLoop(sessions, city.id, KEY, clock=time.clock, sleep=time.sleep, on_phase=bet_when_open).play_round()

    count = placed["count"]
    outcome = "under" if count < 11 else "exact" if count == 11 else "over"
    for choice, (player, bet_id) in placed["bets"].items():
        bet = await _bet_row(sessions, bet_id)
        assert bet.status == ("won" if choice == outcome else "lost")
        assert await _balance(sessions, player) == BALANCE - 1000 + (WIN[choice] if choice == outcome else 0)


async def test_restart_that_voids_a_round_refunds_its_bets(sessions, city):
    round_ = await _round_with_count(sessions, city, 8)
    placed = await _three_bets(sessions, round_.id)
    time = FakeTime()
    time.now = START + timedelta(seconds=15)  # missed its lock while the engine was down
    await _loop(sessions, city, time).recover()
    for player, bet_id in placed.values():
        assert (await _bet_row(sessions, bet_id)).status == "refunded"
        assert await _balance(sessions, player) == BALANCE


async def test_restart_in_playback_pays_the_bets(sessions, city):
    round_ = await _round_with_count(sessions, city, COUNT_FOR["exact"])
    placed = await _three_bets(sessions, round_.id)
    await _to_playback(sessions, round_.id)
    time = FakeTime()
    time.now = START + timedelta(seconds=90)
    await _loop(sessions, city, time).recover()
    player, bet_id = placed["exact"]
    assert (await _bet_row(sessions, bet_id)).status == "won"
    assert await _balance(sessions, player) == BALANCE - 1000 + WIN["exact"]
