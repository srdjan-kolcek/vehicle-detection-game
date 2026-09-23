from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(get_settings().database_url)


async def get_session() -> AsyncIterator[AsyncSession]:
    """One session per request. Handlers commit explicitly; nothing commits on its own."""
    async with async_sessionmaker(get_engine(), expire_on_commit=False)() as session:
        yield session
