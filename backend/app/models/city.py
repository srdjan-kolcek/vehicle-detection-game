from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class City(Base):
    __tablename__ = "cities"
    __table_args__ = (
        CheckConstraint("min_stake > 0 AND max_stake >= min_stake", name="stake_range_valid"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name_key: Mapped[str] = mapped_column(String(128), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lng: Mapped[float] = mapped_column(Float, nullable=False)
    theme_key: Mapped[str] = mapped_column(String(64), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    line_l: Mapped[int] = mapped_column(Integer, nullable=False)
    min_stake: Mapped[int] = mapped_column(BigInteger, nullable=False)
    max_stake: Mapped[int] = mapped_column(BigInteger, nullable=False)


class Clip(Base):
    __tablename__ = "clips"
    __table_args__ = (
        UniqueConstraint("city_id", "file", name="uq_clips_city_id_file"),
        CheckConstraint("count >= 0", name="count_non_negative"),
        Index("ix_clips_city_id_verified", "city_id", "verified"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"), nullable=False)
    angle_id: Mapped[str] = mapped_column(String(64), nullable=False)
    file: Mapped[str] = mapped_column(String(512), nullable=False)
    timeline_file: Mapped[str] = mapped_column(String(512), nullable=False)
    tracks_file: Mapped[str] = mapped_column(String(512), nullable=False)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    # The stored, signed count is the round outcome; it is never recomputed.
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    timeline_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    tracks_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    signature: Mapped[str] = mapped_column(String(128), nullable=False)
    source_dataset: Mapped[str] = mapped_column(String(128), nullable=False)
    licence: Mapped[str] = mapped_column(String(128), nullable=False)
    weights_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    # Placeholder clips are seeded with verified=false and excluded in production.
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))


class PayoutTable(Base):
    __tablename__ = "payout_tables"
    __table_args__ = (
        UniqueConstraint("city_id", "version", name="uq_payout_tables_city_id_version"),
        # At most one active table per city.
        Index(
            "uq_payout_tables_city_id_active",
            "city_id",
            unique=True,
            postgresql_where=text("active"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    p_under: Mapped[Decimal] = mapped_column(Numeric(10, 8), nullable=False)
    p_exact: Mapped[Decimal] = mapped_column(Numeric(10, 8), nullable=False)
    p_over: Mapped[Decimal] = mapped_column(Numeric(10, 8), nullable=False)
    m_under: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)
    m_exact: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)
    m_over: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)
    rtp_under: Mapped[Decimal] = mapped_column(Numeric(10, 8), nullable=False)
    rtp_exact: Mapped[Decimal] = mapped_column(Numeric(10, 8), nullable=False)
    rtp_over: Mapped[Decimal] = mapped_column(Numeric(10, 8), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
