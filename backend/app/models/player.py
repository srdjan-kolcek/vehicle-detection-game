import uuid
from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, created_at_column


class Player(Base):
    __tablename__ = "players"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # NULL for guest players, who cannot log in again.
    password_hash: Mapped[str | None] = mapped_column(String(255))
    locale: Mapped[str] = mapped_column(String(8), nullable=False, server_default="en")
    created_at: Mapped[datetime] = created_at_column()


class WalletAccount(Base):
    """Virtual wallet balance only; the operator wallet keeps its own balance."""

    __tablename__ = "wallet_accounts"
    __table_args__ = (CheckConstraint("balance >= 0", name="balance_non_negative"),)

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), primary_key=True
    )
    # Integer minor units (credits x 100).
    balance: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")


class LedgerEntry(Base):
    """Append-only: a database trigger rejects UPDATE and DELETE."""

    __tablename__ = "ledger_entries"
    __table_args__ = (
        CheckConstraint(
            "type IN ('bet','payout','refund','topup')", name="type_valid"
        ),
        CheckConstraint("balance_after >= 0", name="balance_after_non_negative"),
        Index("ix_ledger_entries_player_id_created_at", "player_id", "created_at"),
        # Makes wallet calls idempotent: one bet/payout/refund entry per bet.
        Index(
            "uq_ledger_entries_bet_id_type",
            "bet_id",
            "type",
            unique=True,
            postgresql_where=text("bet_id IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    bet_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("bets.id"))
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    # Signed minor units: negative for bets, positive for payouts, refunds and top-ups.
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    balance_after: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = created_at_column()
