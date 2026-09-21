import asyncio
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError
from app.models import LedgerEntry
from app.wallet import VirtualWallet
from tests.test_schema import _insert_bet, _seed_round

pytestmark = pytest.mark.asyncio


def _setup(engine, balance: int = 1000) -> tuple[uuid.UUID, uuid.UUID]:
    """A player with a funded wallet and one bet row; returns (player_id, bet_id)."""
    with engine.begin() as conn:
        round_id, player_id = _seed_round(conn)
        bet_id = _insert_bet(conn, round_id, player_id)
        conn.execute(
            text("INSERT INTO wallet_accounts (player_id, balance) VALUES (:p, :b)"),
            {"p": player_id, "b": balance},
        )
    return player_id, bet_id


async def _ledger(session: AsyncSession, player_id: uuid.UUID) -> list[tuple[str, int, int]]:
    rows = await session.scalars(
        select(LedgerEntry).where(LedgerEntry.player_id == player_id).order_by(LedgerEntry.id)
    )
    return [(e.type, e.amount, e.balance_after) for e in rows]


async def test_place_bet_takes_the_stake_and_writes_a_ledger_entry(migrated_engine, session):
    player_id, bet_id = _setup(migrated_engine)
    wallet = VirtualWallet(session)
    await wallet.place_bet(player_id, bet_id, 1, 300)
    await session.commit()
    assert await wallet.get_balance(player_id) == 700
    assert await _ledger(session, player_id) == [("bet", -300, 700)]


async def test_place_bet_is_idempotent(migrated_engine, session):
    player_id, bet_id = _setup(migrated_engine)
    wallet = VirtualWallet(session)
    await wallet.place_bet(player_id, bet_id, 1, 300)
    await wallet.place_bet(player_id, bet_id, 1, 300)
    await session.commit()
    assert await wallet.get_balance(player_id) == 700
    assert len(await _ledger(session, player_id)) == 1


async def test_place_bet_rejects_insufficient_funds(migrated_engine, session):
    player_id, bet_id = _setup(migrated_engine, balance=200)
    wallet = VirtualWallet(session)
    with pytest.raises(AppError) as err:
        await wallet.place_bet(player_id, bet_id, 1, 300)
    assert err.value.code == "INSUFFICIENT_FUNDS"
    await session.rollback()
    assert await wallet.get_balance(player_id) == 200
    assert await _ledger(session, player_id) == []


async def test_place_bet_can_spend_the_whole_balance(migrated_engine, session):
    player_id, bet_id = _setup(migrated_engine, balance=300)
    wallet = VirtualWallet(session)
    await wallet.place_bet(player_id, bet_id, 1, 300)
    await session.commit()
    assert await wallet.get_balance(player_id) == 0


@pytest.mark.parametrize("amount", [0, -100])
async def test_place_bet_rejects_non_positive_stake(migrated_engine, session, amount):
    player_id, bet_id = _setup(migrated_engine)
    with pytest.raises(ValueError):
        await VirtualWallet(session).place_bet(player_id, bet_id, 1, amount)


async def test_settle_credits_the_payout_once(migrated_engine, session):
    player_id, bet_id = _setup(migrated_engine)
    wallet = VirtualWallet(session)
    await wallet.place_bet(player_id, bet_id, 1, 300)
    await wallet.settle(bet_id, 633)
    await wallet.settle(bet_id, 633)
    await session.commit()
    assert await wallet.get_balance(player_id) == 1333
    assert await _ledger(session, player_id) == [("bet", -300, 700), ("payout", 633, 1333)]


async def test_settle_with_zero_payout_records_the_loss(migrated_engine, session):
    player_id, bet_id = _setup(migrated_engine)
    wallet = VirtualWallet(session)
    await wallet.place_bet(player_id, bet_id, 1, 300)
    await wallet.settle(bet_id, 0)
    await session.commit()
    assert await wallet.get_balance(player_id) == 700
    assert await _ledger(session, player_id) == [("bet", -300, 700), ("payout", 0, 700)]


async def test_settle_needs_a_placed_bet(migrated_engine, session):
    _, bet_id = _setup(migrated_engine)
    with pytest.raises(LookupError):
        await VirtualWallet(session).settle(bet_id, 100)


async def test_rollback_refunds_the_stake_once(migrated_engine, session):
    player_id, bet_id = _setup(migrated_engine)
    wallet = VirtualWallet(session)
    await wallet.place_bet(player_id, bet_id, 1, 300)
    await wallet.rollback(bet_id)
    await wallet.rollback(bet_id)
    await session.commit()
    assert await wallet.get_balance(player_id) == 1000
    assert await _ledger(session, player_id) == [("bet", -300, 700), ("refund", 300, 1000)]


async def test_rollback_without_a_stake_does_nothing(migrated_engine, session):
    player_id, bet_id = _setup(migrated_engine)
    wallet = VirtualWallet(session)
    await wallet.rollback(bet_id)
    assert await wallet.get_balance(player_id) == 1000
    assert await _ledger(session, player_id) == []


async def test_a_settled_bet_cannot_be_rolled_back(migrated_engine, session):
    player_id, bet_id = _setup(migrated_engine)
    wallet = VirtualWallet(session)
    await wallet.place_bet(player_id, bet_id, 1, 300)
    await wallet.settle(bet_id, 0)
    with pytest.raises(ValueError):
        await wallet.rollback(bet_id)


async def test_a_rolled_back_bet_cannot_be_settled(migrated_engine, session):
    player_id, bet_id = _setup(migrated_engine)
    wallet = VirtualWallet(session)
    await wallet.place_bet(player_id, bet_id, 1, 300)
    await wallet.rollback(bet_id)
    with pytest.raises(ValueError):
        await wallet.settle(bet_id, 633)


async def test_topup_credits_and_records_a_topup_entry(migrated_engine, session):
    player_id, _ = _setup(migrated_engine, balance=0)
    wallet = VirtualWallet(session)
    assert await wallet.topup(player_id, 100000) == 100000
    assert await wallet.topup(player_id, 100000) == 200000
    await session.commit()
    assert await _ledger(session, player_id) == [
        ("topup", 100000, 100000),
        ("topup", 100000, 200000),
    ]


async def test_unknown_player_has_no_wallet(session, migrated_engine):
    with pytest.raises(LookupError):
        await VirtualWallet(session).get_balance(uuid.uuid4())


async def test_concurrent_bets_cannot_overdraw(migrated_engine, async_engine):
    """Two bets of 100 against a balance of 100: the row lock lets exactly one through."""
    player_id, first_bet = _setup(migrated_engine, balance=100)
    with migrated_engine.begin() as conn:
        second_round = conn.execute(
            text(
                "INSERT INTO rounds (city_id, clip_id, status, payout_table_id, betting_opens_at, "
                "locks_at, playback_starts_at) SELECT city_id, clip_id, status, payout_table_id, "
                "now(), now(), now() FROM rounds LIMIT 1 RETURNING id"
            )
        ).scalar_one()
        second_bet = _insert_bet(conn, second_round, player_id)

    async def attempt(bet_id: uuid.UUID) -> str:
        async with AsyncSession(async_engine) as s:
            try:
                await VirtualWallet(s).place_bet(player_id, bet_id, 1, 100)
                await s.commit()
                return "ok"
            except AppError as e:
                return e.code

    results = await asyncio.gather(attempt(first_bet), attempt(second_bet))
    assert sorted(results) == ["INSUFFICIENT_FUNDS", "ok"]
    async with AsyncSession(async_engine) as s:
        assert await VirtualWallet(s).get_balance(player_id) == 0
