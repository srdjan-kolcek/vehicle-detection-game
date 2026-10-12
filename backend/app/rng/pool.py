import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.media.signing import clip_signature_valid
from app.models import City, Clip, Round
from app.rng.draw import Draw, cooldown_size, draw_clip

log = logging.getLogger(__name__)


async def load_pool(session: AsyncSession, city: City, signing_key: str) -> list[Clip]:
    """The city's verified clips whose signature checks out, sorted by id."""
    clips = await session.scalars(
        select(Clip).where(Clip.city_id == city.id, Clip.verified.is_(True)).order_by(Clip.id)
    )
    pool = []
    for clip in clips:
        if clip_signature_valid(clip, city.slug, signing_key):
            pool.append(clip)
        else:
            log.warning("clip %s (%s) has a bad signature and is excluded from draws", clip.id, clip.file)
    return pool


async def recent_clip_ids(session: AsyncSession, city_id: int, limit: int) -> list[int]:
    """Clip ids of the city's last `limit` rounds (any status, voided included), newest first."""
    if limit <= 0:
        return []
    rows = await session.scalars(
        select(Round.clip_id).where(Round.city_id == city_id).order_by(Round.id.desc()).limit(limit)
    )
    return list(rows)


async def draw_for_city(session: AsyncSession, city: City, signing_key: str) -> Draw:
    """Draw the next clip for a city. Raises EmptyPoolError when nothing is drawable."""
    pool = await load_pool(session, city, signing_key)
    recent = await recent_clip_ids(session, city.id, cooldown_size(len(pool)))
    return draw_clip([c.id for c in pool], recent)
