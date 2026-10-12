"""Bet placement (needs TEST_DATABASE_URL)."""
import asyncio
import uuid
from datetime import timedelta
from decimal import Decimal

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import AppError
from app.db import get_session
from app.engine.bets import place_bet
from app.engine.loop import utc_now
from app.engine.rounds import advance, void_round
from app.main import app
from app.models import Bet, LedgerEntry, Player, WalletAccount
from app.wallet.virtual import VirtualWallet
from tests.test_engine_loop import START, _open, _setup_city

pytestmark = pytest.mark.asyncio

BALANCE = 100_000  # the city allows stakes of 100..10000


@pytest_asyncio.fixture()
async def sessions(async_engine):
    return async_sessionmaker(async_engine, expire_on_commit=False)


@pytest_asyncio.fixture()
async def city(session):
    return await _setup_city(session)


@pytest_asyncio.fixture()
async def round_(sessions, city):
    return await _open(sessions, city)  # betting until START + 15 s


async def _player(sessions, balance=BALANCE) -> uuid.UUID:
    player_id = uuid.uuid4()
    async with sessions() as s:
        s.add(Player(id=player_id, username=f"p-{player_id.hex[:8]}"))
        await s.flush()
        s.add(WalletAccount(player_id=player_id, balance=0))
        await s.flush()
        if balance:
            await VirtualWallet(s).topup(player_id, balance)
        await s.commit()
    return player_id


async def _bet(sessions, player_id, round_id, *, bet_id=None, choice="under", stake=1000, at=5):
    """Place a bet in its own transaction, as the endpoint does."""
    async with sessions() as s:
        bet, created = await place_bet(
            s, VirtualWallet(s), player_id=player_id, round_id=round_id, bet_id=bet_id or uuid.uuid4(),
            choice=choice, stake=stake, now=START + timedelta(seconds=at),
        )
        await s.commit()
    return bet, created


async def _balance(sessions, player_id) -> int:
    async with sessions() as s:
        return await VirtualWallet(s).get_balance(player_id)


async def _bets(sessions, player_id) -> list[Bet]:
    async with sessions() as s:
        return list(await s.scalars(select(Bet).where(Bet.player_id == player_id)))


async def _code(coro) -> str:
    with pytest.raises(AppError) as exc:
        await coro
    return exc.value.code


# --- placing ---

async def test_bet_takes_the_stake_at_the_rounds_odds(sessions, round_):
    player = await _player(sessions)
    bet, created = await _bet(sessions, player, round_.id, choice="over", stake=2500)
    assert created
    assert (bet.choice, bet.stake, bet.status, bet.payout) == ("over", 2500, "open", 0)
    assert bet.multiplier == Decimal("2.25")  # m_over of the round's payout table
    assert await _balance(sessions, player) == BALANCE - 2500
    async with sessions() as s:
        entry = await s.scalar(select(LedgerEntry).where(LedgerEntry.bet_id == bet.id))
    assert (entry.type, entry.amount, entry.balance_after) == ("bet", -2500, BALANCE - 2500)


@pytest.mark.parametrize(("choice", "multiplier"), [("under", "1.80"), ("exact", "8.00"), ("over", "2.25")])
async def test_each_choice_gets_its_own_multiplier(sessions, round_, choice, multiplier):
    bet, _ = await _bet(sessions, await _player(sessions), round_.id, choice=choice)
    assert bet.multiplier == Decimal(multiplier)


async def test_unknown_choice_is_rejected(sessions, round_):
    with pytest.raises(ValueError):
        await _bet(sessions, await _player(sessions), round_.id, choice="draw")


# --- one bet per player per round, final ---

async def test_second_bet_in_the_same_round_is_refused(sessions, round_):
    player = await _player(sessions)
    await _bet(sessions, player, round_.id, choice="under")
    assert await _code(_bet(sessions, player, round_.id, choice="over")) == "BET_ALREADY_PLACED"
    assert await _code(_bet(sessions, player, round_.id, choice="under")) == "BET_ALREADY_PLACED"
    assert len(await _bets(sessions, player)) == 1
    assert await _balance(sessions, player) == BALANCE - 1000


async def test_other_players_and_other_rounds_are_independent(sessions, city, round_):
    first, second = await _player(sessions), await _player(sessions)
    await _bet(sessions, first, round_.id)
    await _bet(sessions, second, round_.id)
    async with sessions() as s:
        await void_round(s, round_.id, START)
        await s.commit()
    next_round = await _open(sessions, city, at=START + timedelta(seconds=60))
    await _bet(sessions, first, next_round.id, at=65)
    assert len(await _bets(sessions, first)) == 2


async def test_parallel_bets_from_one_player_let_exactly_one_through(sessions, round_):
    player = await _player(sessions)
    results = await asyncio.gather(
        *(_bet(sessions, player, round_.id, choice=c) for c in ("under", "exact", "over")),
        return_exceptions=True,
    )
    placed = [r for r in results if not isinstance(r, Exception)]
    refused = [r for r in results if isinstance(r, AppError)]
    assert len(placed) == 1
    assert [e.code for e in refused] == ["BET_ALREADY_PLACED"] * 2
    assert await _balance(sessions, player) == BALANCE - 1000


# --- resubmits ---

async def test_resubmitting_the_same_bet_id_returns_the_bet_unchanged(sessions, round_):
    player, bet_id = await _player(sessions), uuid.uuid4()
    first, created = await _bet(sessions, player, round_.id, bet_id=bet_id, choice="under", stake=1000)
    again, created_again = await _bet(sessions, player, round_.id, bet_id=bet_id, choice="over", stake=5000, at=30)
    assert created and not created_again
    assert (again.id, again.choice, again.stake) == (first.id, "under", 1000)  # even after the lock
    assert await _balance(sessions, player) == BALANCE - 1000


async def test_bet_id_used_by_another_player_is_a_conflict(sessions, round_):
    bet_id = uuid.uuid4()
    await _bet(sessions, await _player(sessions), round_.id, bet_id=bet_id)
    other = await _player(sessions)
    assert await _code(_bet(sessions, other, round_.id, bet_id=bet_id)) == "BET_ID_CONFLICT"
    assert await _balance(sessions, other) == BALANCE


# --- round state and limits ---

async def test_bet_at_or_after_locks_at_is_refused_even_before_the_engine_locks(sessions, round_):
    player = await _player(sessions)
    assert await _code(_bet(sessions, player, round_.id, at=15)) == "BET_WINDOW_CLOSED"
    await _bet(sessions, player, round_.id, at=14.999)


@pytest.mark.parametrize("status", ["LOCKED", "PLAYBACK"])
async def test_bet_in_a_later_phase_is_refused(sessions, round_, status):
    async with sessions() as s:
        await advance(s, round_.id, "BETTING", "LOCKED")
        if status == "PLAYBACK":
            await advance(s, round_.id, "LOCKED", "PLAYBACK")
        await s.commit()
    assert await _code(_bet(sessions, await _player(sessions), round_.id)) == "BET_WINDOW_CLOSED"


async def test_bet_on_a_voided_round_is_refused(sessions, round_):
    async with sessions() as s:
        await void_round(s, round_.id, START)
        await s.commit()
    assert await _code(_bet(sessions, await _player(sessions), round_.id)) == "ROUND_VOIDED"


async def test_bet_on_an_unknown_round_is_refused(sessions, round_):
    assert await _code(_bet(sessions, await _player(sessions), round_.id + 999)) == "ROUND_NOT_FOUND"


@pytest.mark.parametrize("stake", [99, 10_001])
async def test_stake_outside_the_city_range_is_refused(sessions, round_, stake):
    player = await _player(sessions)
    assert await _code(_bet(sessions, player, round_.id, stake=stake)) == "STAKE_OUT_OF_RANGE"
    assert await _bets(sessions, player) == []


@pytest.mark.parametrize("stake", [100, 10_000])
async def test_stake_at_the_city_limits_is_accepted(sessions, round_, stake):
    await _bet(sessions, await _player(sessions), round_.id, stake=stake)


async def test_insufficient_funds_keeps_nothing(sessions, round_):
    player = await _player(sessions, balance=500)
    assert await _code(_bet(sessions, player, round_.id, stake=1000)) == "INSUFFICIENT_FUNDS"
    assert await _bets(sessions, player) == []
    assert await _balance(sessions, player) == 500


async def test_locking_waits_for_a_bet_being_saved(sessions, round_):
    """The engine's BETTING -> LOCKED update queues behind an open bet transaction."""
    player = await _player(sessions)
    async with sessions() as bet_session:
        await place_bet(
            bet_session, VirtualWallet(bet_session), player_id=player, round_id=round_.id,
            bet_id=uuid.uuid4(), choice="under", stake=1000, now=START + timedelta(seconds=14),
        )

        async def lock():
            async with sessions() as s:
                await advance(s, round_.id, "BETTING", "LOCKED")
                await s.commit()

        locking = asyncio.create_task(lock())
        await asyncio.sleep(0.3)
        assert not locking.done()
        await bet_session.commit()
    await asyncio.wait_for(locking, timeout=5)
    assert len(await _bets(sessions, player)) == 1


# --- endpoint ---

@pytest_asyncio.fixture()
async def client(async_engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://unused")
    monkeypatch.setenv("DEMO_START_CREDITS", "1000")

    async def override_session():
        async with AsyncSession(async_engine, expire_on_commit=False) as s:
            yield s

    app.dependency_overrides[get_session] = override_session
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def _token(client) -> str:
    return (await client.post("/api/auth/guest")).json()["token"]


async def test_endpoint_places_and_replays_a_bet(client, sessions, city):
    round_ = await _open(sessions, city, at=utc_now())
    headers = {"Authorization": f"Bearer {await _token(client)}"}
    body = {"bet_id": str(uuid.uuid4()), "choice": "exact", "stake": 1000}

    first = await client.post(f"/api/rounds/{round_.id}/bet", json=body, headers=headers)
    assert first.status_code == 201
    assert first.json() == {
        "bet_id": body["bet_id"], "round_id": round_.id, "choice": "exact", "stake": 1000,
        "multiplier": "8.00", "status": "open", "balance": 99_000,
    }
    again = await client.post(f"/api/rounds/{round_.id}/bet", json=body, headers=headers)
    assert again.status_code == 200
    assert again.json() == first.json()

    second = await client.post(
        f"/api/rounds/{round_.id}/bet", json={**body, "bet_id": str(uuid.uuid4())}, headers=headers
    )
    assert second.status_code == 409
    assert second.json() == {"code": "BET_ALREADY_PLACED", "params": {"round_id": round_.id}}


async def test_endpoint_error_shapes(client, sessions, city):
    round_ = await _open(sessions, city, at=utc_now())
    headers = {"Authorization": f"Bearer {await _token(client)}"}
    body = {"bet_id": str(uuid.uuid4()), "choice": "under", "stake": 50}

    no_auth = await client.post(f"/api/rounds/{round_.id}/bet", json=body)
    assert (no_auth.status_code, no_auth.json()["code"]) == (401, "NOT_AUTHENTICATED")
    low = await client.post(f"/api/rounds/{round_.id}/bet", json=body, headers=headers)
    assert (low.status_code, low.json()) == (
        400, {"code": "STAKE_OUT_OF_RANGE", "params": {"min_stake": 100, "max_stake": 10000}}
    )
    missing = await client.post(f"/api/rounds/{round_.id + 999}/bet", json={**body, "stake": 100}, headers=headers)
    assert (missing.status_code, missing.json()["code"]) == (404, "ROUND_NOT_FOUND")
    bad_choice = await client.post(f"/api/rounds/{round_.id}/bet", json={**body, "choice": "draw"}, headers=headers)
    assert bad_choice.status_code == 422
    float_stake = await client.post(f"/api/rounds/{round_.id}/bet", json={**body, "stake": 100.5}, headers=headers)
    assert float_stake.status_code == 422
