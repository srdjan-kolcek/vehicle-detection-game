import pytest
from sqlalchemy import func, select, update

from app import seed
from app.core.config import Settings
from app.core.security import verify_password
from app.models import City, LedgerEntry, Player, WalletAccount
from app.seed import DEMO_USERNAMES, seed_cities, seed_players

pytestmark = pytest.mark.asyncio

SETTINGS = Settings(
    database_url="unused",
    jwt_secret="test",
    jwt_ttl_s=3600,
    demo_start_credits=1000,
    demo_player_password="demo-password",
    rehydrate_cooldown_s=0,
)


async def _count(session, model) -> int:
    return await session.scalar(select(func.count()).select_from(model))


async def test_seed_adds_the_five_cities_with_las_vegas_first(session):
    assert await seed_cities(session) == 5
    await session.commit()
    slugs = list(await session.scalars(select(City.slug).order_by(City.id)))
    assert slugs == ["las_vegas", "belgrade", "berlin", "moscow", "tokyo"]
    vegas = await session.scalar(select(City).where(City.slug == "las_vegas"))
    assert (vegas.name_key, vegas.theme_key, vegas.active) == ("cities.las_vegas", "vegas", True)


async def test_seed_adds_demo_players_with_starting_credits(session):
    assert await seed_players(session, SETTINGS) == 3
    await session.commit()
    for username in DEMO_USERNAMES:
        player = await session.scalar(select(Player).where(Player.username == username))
        assert verify_password(player.password_hash, "demo-password")
        balance = await session.scalar(
            select(WalletAccount.balance).where(WalletAccount.player_id == player.id)
        )
        assert balance == 100_000
        entries = await session.scalars(
            select(LedgerEntry).where(LedgerEntry.player_id == player.id)
        )
        assert [(e.type, e.amount, e.balance_after) for e in entries] == [
            ("topup", 100_000, 100_000)
        ]


async def test_seeding_twice_adds_nothing(session):
    await seed_cities(session)
    await seed_players(session, SETTINGS)
    await session.commit()
    assert await seed_cities(session) == 0
    assert await seed_players(session, SETTINGS) == 0
    await session.commit()
    assert await _count(session, City) == 5
    assert await _count(session, Player) == 3
    assert await _count(session, WalletAccount) == 3
    assert await _count(session, LedgerEntry) == 3


async def test_seeding_again_keeps_balances_and_edited_cities(session):
    await seed_cities(session)
    await seed_players(session, SETTINGS)
    await session.execute(update(WalletAccount).values(balance=42))
    await session.execute(update(City).where(City.slug == "berlin").values(line_l=17))
    await session.commit()

    await seed_cities(session)
    await seed_players(session, SETTINGS)
    await session.commit()

    balances = set(await session.scalars(select(WalletAccount.balance)))
    assert balances == {42}
    assert await session.scalar(select(City.line_l).where(City.slug == "berlin")) == 17


async def test_seed_only_adds_a_newly_listed_demo_player(session, monkeypatch):
    await seed_players(session, SETTINGS)
    await session.execute(update(WalletAccount).values(balance=0))
    await session.commit()

    monkeypatch.setattr(seed, "DEMO_USERNAMES", DEMO_USERNAMES + ("demo4",))
    assert await seed_players(session, SETTINGS) == 1
    await session.commit()

    assert await _count(session, Player) == 4
    balances = sorted(await session.scalars(select(WalletAccount.balance)))
    assert balances == [0, 0, 0, 100_000]
