"""Store new target RTPs and give every active city a payout table built for them. Never commits."""
from dataclasses import dataclass

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AppConfig, City, PayoutTable
from app.payout import TARGET_RTP_KEY, TargetRtp, build_table
from app.rng.pool import load_pool


@dataclass(frozen=True)
class CityTable:
    slug: str
    line: int
    pool_size: int
    row: dict
    version: int | None  # the new version, or None when the active table already matches


async def load_targets(session: AsyncSession) -> TargetRtp:
    """Targets from app_config, or the defaults when the key is missing. Raises ValueError."""
    value = await session.scalar(select(AppConfig.value).where(AppConfig.key == TARGET_RTP_KEY))
    return TargetRtp() if value is None else TargetRtp.from_config(value)


async def retarget(session: AsyncSession, targets: TargetRtp, signing_key: str) -> list[CityTable]:
    """Raises ValueError, naming the city, when a pool cannot be priced for these targets."""
    targets.check()
    await session.execute(
        insert(AppConfig)
        .values(key=TARGET_RTP_KEY, value=targets.as_config())
        .on_conflict_do_update(index_elements=["key"], set_={"value": targets.as_config()})
    )
    results = []
    cities = await session.scalars(select(City).where(City.active.is_(True)).order_by(City.id))
    for city in cities:
        pool = await load_pool(session, city, signing_key)
        try:
            row = build_table([clip.count for clip in pool], city.line_l, targets)
        except ValueError as exc:
            raise ValueError(f"{city.slug}: {exc}") from None
        active = await session.scalar(
            select(PayoutTable).where(PayoutTable.city_id == city.id, PayoutTable.active.is_(True))
        )
        version = None
        if active is None or any(getattr(active, column) != value for column, value in row.items()):
            version = (await session.scalar(
                select(func.coalesce(func.max(PayoutTable.version), 0)).where(PayoutTable.city_id == city.id)
            )) + 1
            await session.execute(
                update(PayoutTable).where(PayoutTable.city_id == city.id, PayoutTable.active.is_(True))
                .values(active=False)
            )
            session.add(PayoutTable(city_id=city.id, version=version, active=True, **row))
            await session.flush()
        results.append(CityTable(city.slug, city.line_l, len(pool), row, version))
    return results
