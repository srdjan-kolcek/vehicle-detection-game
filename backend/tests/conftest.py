import asyncio
import os
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

BACKEND_DIR = Path(__file__).resolve().parents[1]

# psycopg's async mode cannot run on the default Windows event loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    return url


@pytest.fixture()
def migrated_engine(database_url: str, monkeypatch: pytest.MonkeyPatch) -> Engine:
    """A clean database with all migrations applied; wiped afterwards."""
    monkeypatch.setenv("DATABASE_URL", database_url)
    engine = create_engine(database_url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    command.upgrade(cfg, "head")
    yield engine
    engine.dispose()


@pytest_asyncio.fixture()
async def async_engine(migrated_engine: Engine, database_url: str):
    engine = create_async_engine(database_url)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture()
async def session(async_engine):
    async with AsyncSession(async_engine, expire_on_commit=False) as s:
        yield s
