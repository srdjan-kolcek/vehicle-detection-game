import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError
from app.models import LedgerEntry, WalletAccount
from app.wallet.base import Money


class VirtualWallet:
    """Demo wallet backed by wallet_accounts and the append-only ledger.

    Never commits: the caller owns the transaction, so a bet row, its stake and
    the balance change succeed or fail together. The bet row must already exist
    in the session's transaction when place_bet runs (ledger.bet_id is a foreign key).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_balance(self, player_id: uuid.UUID) -> Money:
        balance = await self._session.scalar(
            select(WalletAccount.balance).where(WalletAccount.player_id == player_id)
        )
        if balance is None:
            raise LookupError(f"no wallet account for player {player_id}")
        return balance

    async def place_bet(
        self, player_id: uuid.UUID, bet_id: uuid.UUID, round_id: int, amount: Money
    ) -> None:
        # round_id is part of the adapter contract (an operator wallet needs it);
        # the virtual ledger identifies the bet by bet_id alone.
        if amount <= 0:
            raise ValueError("stake must be positive")
        account = await self._lock_account(player_id)
        if await self._entry(bet_id, "bet") is not None:
            return
        if account.balance < amount:
            raise AppError("INSUFFICIENT_FUNDS", balance=account.balance, stake=amount)
        self._apply(account, "bet", -amount, bet_id)

    async def settle(self, bet_id: uuid.UUID, payout: Money) -> None:
        if payout < 0:
            raise ValueError("payout cannot be negative")
        stake_entry = await self._stake_entry(bet_id)
        account = await self._lock_account(stake_entry.player_id)
        if await self._entry(bet_id, "payout") is not None:
            return
        if await self._entry(bet_id, "refund") is not None:
            raise ValueError("bet was rolled back and cannot be settled")
        # A zero payout (lost bet) is still recorded: every settled bet has exactly one payout row.
        self._apply(account, "payout", payout, bet_id)

    async def rollback(self, bet_id: uuid.UUID) -> None:
        stake_entry = await self._entry(bet_id, "bet")
        if stake_entry is None:
            return  # the stake never left the wallet, so there is nothing to give back
        account = await self._lock_account(stake_entry.player_id)
        if await self._entry(bet_id, "refund") is not None:
            return
        if await self._entry(bet_id, "payout") is not None:
            raise ValueError("bet is already settled and cannot be rolled back")
        self._apply(account, "refund", -stake_entry.amount, bet_id)

    async def topup(self, player_id: uuid.UUID, amount: Money) -> Money:
        """Credit the player and record a topup entry. Not idempotent: each call is a new top-up."""
        if amount <= 0:
            raise ValueError("top-up must be positive")
        account = await self._lock_account(player_id)
        self._apply(account, "topup", amount, None)
        return account.balance

    async def rehydrate(self, player_id: uuid.UUID, amount: Money, cooldown_s: int = 0) -> Money:
        """Demo-only top-up, gated by an optional cooldown since the player's last one."""
        if amount <= 0:
            raise ValueError("rehydrate amount must be positive")
        account = await self._lock_account(player_id)
        if cooldown_s > 0:
            last_topup = await self._session.scalar(
                select(LedgerEntry.created_at)
                .where(LedgerEntry.player_id == player_id, LedgerEntry.type == "topup")
                .order_by(LedgerEntry.id.desc())
                .limit(1)
            )
            if last_topup is not None:
                elapsed = (datetime.now(timezone.utc) - last_topup).total_seconds()
                if elapsed < cooldown_s:
                    raise AppError("REHYDRATE_COOLDOWN", retry_after_s=int(cooldown_s - elapsed))
        self._apply(account, "topup", amount, None)
        return account.balance

    async def _lock_account(self, player_id: uuid.UUID) -> WalletAccount:
        """Row lock, so concurrent calls for one player queue up and never overdraw."""
        account = await self._session.scalar(
            select(WalletAccount)
            .where(WalletAccount.player_id == player_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if account is None:
            raise LookupError(f"no wallet account for player {player_id}")
        return account

    async def _entry(self, bet_id: uuid.UUID, type_: str) -> LedgerEntry | None:
        return await self._session.scalar(
            select(LedgerEntry).where(LedgerEntry.bet_id == bet_id, LedgerEntry.type == type_)
        )

    async def _stake_entry(self, bet_id: uuid.UUID) -> LedgerEntry:
        entry = await self._entry(bet_id, "bet")
        if entry is None:
            raise LookupError(f"no stake was placed for bet {bet_id}")
        return entry

    def _apply(
        self, account: WalletAccount, type_: str, amount: Money, bet_id: uuid.UUID | None
    ) -> None:
        account.balance += amount
        self._session.add(
            LedgerEntry(
                player_id=account.player_id,
                bet_id=bet_id,
                type=type_,
                amount=amount,
                balance_after=account.balance,
            )
        )
