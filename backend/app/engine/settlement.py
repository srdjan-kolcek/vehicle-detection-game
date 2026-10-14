"""Bet settlement (doc 03): pay the winning choice, close the losing ones, refund every stake of a voided round.

Runs inside the round's transaction and never commits. The caller holds the round's row lock, so one round is
never settled twice at once. Only open bets are touched and every wallet call is idempotent, so repeating a
step changes nothing.
"""
from decimal import ROUND_FLOOR, Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Bet
from app.wallet.base import Money, WalletAdapter


def payout_for(stake: Money, multiplier: Decimal) -> Money:
    """What a winning bet returns, stake included, rounded down to whole minor units."""
    return int((Decimal(stake) * multiplier).to_integral_value(rounding=ROUND_FLOOR))


async def _open_bets(session: AsyncSession, round_id: int) -> list[Bet]:
    # Player order: rounds of different cities settling at once lock wallet accounts in the same order,
    # so they cannot deadlock on players who bet in both.
    return list(await session.scalars(
        select(Bet).where(Bet.round_id == round_id, Bet.status == "open").order_by(Bet.player_id)
    ))


async def settle_bets(session: AsyncSession, wallet: WalletAdapter, round_id: int, outcome: str) -> None:
    """Mark each open bet won or lost and pay it. A lost bet is paid 0, so every settled bet has a payout row."""
    for bet in await _open_bets(session, round_id):
        won = bet.choice == outcome
        bet.status = "won" if won else "lost"
        bet.payout = payout_for(bet.stake, bet.multiplier) if won else 0
        await wallet.settle(bet.id, bet.payout)
    await session.flush()


async def refund_bets(session: AsyncSession, wallet: WalletAdapter, round_id: int) -> None:
    """Give every open bet its stake back."""
    for bet in await _open_bets(session, round_id):
        bet.status = "refunded"
        await wallet.rollback(bet.id)
    await session.flush()
