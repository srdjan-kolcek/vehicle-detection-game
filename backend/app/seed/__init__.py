import uuid
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.security import hash_password
from app.models import City, Player, WalletAccount
from app.wallet import VirtualWallet

CITIES_FILE = Path(__file__).with_name("cities.yaml")
DEMO_USERNAMES = ("demo1", "demo2", "demo3")


async def seed_cities(session: AsyncSession) -> int:
    """Add the cities that are missing. Existing rows are left as they are. Returns how many were added."""
    rows = yaml.safe_load(CITIES_FILE.read_text(encoding="utf-8"))["cities"]
    result = await session.execute(
        insert(City).values(rows).on_conflict_do_nothing(index_elements=["slug"]).returning(City.id)
    )
    return len(result.all())


async def seed_players(session: AsyncSession, settings: Settings) -> int:
    """Add the demo players that are missing, each with the starting credits.

    A player that already exists is skipped entirely, so its balance is never reset.
    Returns how many were added.
    """
    wallet = VirtualWallet(session)
    added = 0
    for username in DEMO_USERNAMES:
        if await session.scalar(select(Player.id).where(Player.username == username)) is not None:
            continue
        player_id = await session.scalar(
            insert(Player)
            .values(
                id=uuid.uuid4(),
                username=username,
                password_hash=hash_password(settings.demo_player_password),
            )
            .on_conflict_do_nothing(index_elements=["username"])
            .returning(Player.id)
        )
        if player_id is None:  # another seed run created it first
            continue
        session.add(WalletAccount(player_id=player_id, balance=0))
        await session.flush()
        await wallet.topup(player_id, settings.demo_start_credits * 100)
        added += 1
    return added
