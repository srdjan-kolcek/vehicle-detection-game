import asyncio
import sys

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db import get_engine
from app.seed import seed_cities, seed_players


async def main() -> None:
    settings = get_settings()
    engine = get_engine()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            cities = await seed_cities(session)
            players = await seed_players(session, settings)
            await session.commit()
    finally:
        await engine.dispose()
    print(f"Seed done: {cities} cities added, {players} demo players added.")


# psycopg's async mode cannot run on the default Windows event loop, so ask for the selector loop there.
with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
    runner.run(main())
