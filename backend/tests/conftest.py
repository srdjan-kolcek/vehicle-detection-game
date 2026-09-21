import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

BACKEND_DIR = Path(__file__).resolve().parents[1]


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
