"""Bet placement (doc 03): one bet per player per round, final once placed. Never commits."""
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError
from app.models import Bet, City, PayoutTable, Player, Round
from app.wallet.base import Money, WalletAdapter

CHOICES = ("under", "exact", "over")


async def place_bet(
    session: AsyncSession,
    wallet: WalletAdapter,
    *,
    player_id: uuid.UUID,
    round_id: int,
    bet_id: uuid.UUID,
    choice: str,
    stake: Money,
    now: datetime,
) -> tuple[Bet, bool]:
    """Place a bet and take the stake, all in the caller's transaction. Returns (bet, created).

    Resubmitting the same bet_id for the same player and round returns the existing bet unchanged,
    even after the round has locked, so a client can safely retry.
    """
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {CHOICES}")

    # Queue this player's requests, so "already has a bet" cannot change between the check and the insert.
    await session.execute(
        select(Player.id).where(Player.id == player_id).with_for_update(key_share=True)
    )

    existing = await session.get(Bet, bet_id)
    if existing is not None:
        if existing.player_id == player_id and existing.round_id == round_id:
            return existing, False
        raise AppError("BET_ID_CONFLICT")

    # A shared lock: bets do not wait for each other, but the engine's BETTING -> LOCKED update waits
    # for bets still being saved, so none lands after the lock.
    round_ = await session.scalar(select(Round).where(Round.id == round_id).with_for_update(read=True))
    if round_ is None:
        raise AppError("ROUND_NOT_FOUND", round_id=round_id)
    if round_.status == "VOIDED":
        raise AppError("ROUND_VOIDED", round_id=round_id)
    if round_.status != "BETTING" or now >= round_.locks_at:
        raise AppError("BET_WINDOW_CLOSED", round_id=round_id)

    if await session.scalar(select(Bet.id).where(Bet.round_id == round_id, Bet.player_id == player_id)):
        raise AppError("BET_ALREADY_PLACED", round_id=round_id)

    city = await session.get(City, round_.city_id)
    if not city.min_stake <= stake <= city.max_stake:
        raise AppError("STAKE_OUT_OF_RANGE", min_stake=city.min_stake, max_stake=city.max_stake)

    # The odds of the payout table pinned on the round, so the player gets what was shown when betting.
    table = await session.get(PayoutTable, round_.payout_table_id)
    bet = Bet(
        id=bet_id,
        round_id=round_id,
        player_id=player_id,
        choice=choice,
        stake=stake,
        multiplier=getattr(table, f"m_{choice}"),
        status="open",
    )
    session.add(bet)
    try:
        await session.flush()
    except IntegrityError as exc:  # a race the player lock does not cover: the same bet_id from another player
        if "uq_bets_round_id_player_id" in str(exc.orig):
            raise AppError("BET_ALREADY_PLACED", round_id=round_id) from exc
        raise AppError("BET_ID_CONFLICT") from exc
    await wallet.place_bet(player_id, bet_id, round_id, stake)
    return bet, True
