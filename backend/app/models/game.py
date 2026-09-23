import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, created_at_column


class Round(Base):
    __tablename__ = "rounds"
    __table_args__ = (
        CheckConstraint(
            "status IN ('BETTING','LOCKED','PLAYBACK','SETTLED','VOIDED')", name="status_valid"
        ),
        CheckConstraint(
            "result_outcome IS NULL OR result_outcome IN ('under','exact','over')",
            name="result_outcome_valid",
        ),
        Index("ix_rounds_city_id_status", "city_id", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"), nullable=False)
    clip_id: Mapped[int] = mapped_column(ForeignKey("clips.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    payout_table_id: Mapped[int] = mapped_column(ForeignKey("payout_tables.id"), nullable=False)
    betting_opens_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    locks_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    playback_starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result_count: Mapped[int | None] = mapped_column(Integer)
    result_outcome: Mapped[str | None] = mapped_column(String(8))


class Bet(Base):
    __tablename__ = "bets"
    __table_args__ = (
        # One bet per player per round (prevents hedging).
        UniqueConstraint("round_id", "player_id", name="uq_bets_round_id_player_id"),
        CheckConstraint("choice IN ('under','exact','over')", name="choice_valid"),
        CheckConstraint("status IN ('open','won','lost','refunded')", name="status_valid"),
        CheckConstraint("stake > 0", name="stake_positive"),
        CheckConstraint("payout >= 0", name="payout_non_negative"),
        Index("ix_bets_round_id", "round_id"),
        Index("ix_bets_player_id_created_at", "player_id", "created_at"),
    )

    # Client-generated UUID; makes submission idempotent.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    round_id: Mapped[int] = mapped_column(ForeignKey("rounds.id"), nullable=False)
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"), nullable=False)
    choice: Mapped[str] = mapped_column(String(8), nullable=False)
    stake: Mapped[int] = mapped_column(BigInteger, nullable=False)
    multiplier: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="open")
    payout: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    created_at: Mapped[datetime] = created_at_column()


class RngAudit(Base):
    """Append-only: a database trigger rejects UPDATE and DELETE."""

    __tablename__ = "rng_audit"
    __table_args__ = (Index("ix_rng_audit_round_id", "round_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(ForeignKey("rounds.id"), nullable=False)
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"), nullable=False)
    pool_size: Mapped[int] = mapped_column(Integer, nullable=False)
    cooldown_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=False)
    chosen_clip_id: Mapped[int] = mapped_column(ForeignKey("clips.id"), nullable=False)
    # Raw CSPRNG output as a decimal string (may exceed 64 bits).
    raw_random: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = created_at_column()


class AppConfig(Base):
    __tablename__ = "app_config"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict | list | int | float | str | bool] = mapped_column(JSONB, nullable=False)
