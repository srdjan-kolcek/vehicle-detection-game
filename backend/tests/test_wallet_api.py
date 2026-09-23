import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.main import app

pytestmark = pytest.mark.asyncio

PASSWORD = "correct horse"


@pytest_asyncio.fixture()
async def client(async_engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://unused")
    monkeypatch.setenv("DEMO_START_CREDITS", "1000")

    async def override_session():
        async with AsyncSession(async_engine, expire_on_commit=False) as s:
            yield s

    app.dependency_overrides[get_session] = override_session
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()


async def register(client, username="alice", password=PASSWORD):
    return await client.post("/api/auth/register", json={"username": username, "password": password})


async def test_rehydrate_credits_demo_start_credits(client):
    token = (await register(client)).json()["token"]
    resp = await client.post(
        "/api/wallet/rehydrate", headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"balance": 200_000}


async def test_rehydrate_can_be_called_repeatedly_with_no_cooldown(client):
    token = (await register(client)).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    await client.post("/api/wallet/rehydrate", headers=headers)
    resp = await client.post("/api/wallet/rehydrate", headers=headers)
    assert resp.json() == {"balance": 300_000}


async def test_rehydrate_requires_authentication(client):
    resp = await client.post("/api/wallet/rehydrate")
    assert resp.status_code == 401
    assert resp.json()["code"] == "NOT_AUTHENTICATED"


async def test_rehydrate_enforces_cooldown(client, monkeypatch):
    """The cooldown is read fresh per request, so it can be turned on between two calls here."""
    token = (await register(client)).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    first = await client.post("/api/wallet/rehydrate", headers=headers)
    assert first.status_code == 200
    monkeypatch.setenv("REHYDRATE_COOLDOWN_S", "60")
    second = await client.post("/api/wallet/rehydrate", headers=headers)
    assert second.status_code == 429
    assert second.json()["code"] == "REHYDRATE_COOLDOWN"
