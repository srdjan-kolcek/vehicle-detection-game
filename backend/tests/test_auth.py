import uuid

import httpx
import pytest
import pytest_asyncio
from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import current_player
from app.core.security import create_token
from app.db import get_session
from app.main import app
from app.models import LedgerEntry, Player

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


async def test_register_creates_player_with_starting_credits(client, session):
    resp = await register(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["username"] == "alice"
    assert body["is_guest"] is False
    assert body["balance"] == 100_000
    assert body["token"]

    player = await session.scalar(select(Player).where(Player.username == "alice"))
    assert player.password_hash.startswith("$argon2")
    entries = (await session.scalars(select(LedgerEntry).where(LedgerEntry.player_id == player.id))).all()
    assert [(e.type, e.amount, e.balance_after) for e in entries] == [("topup", 100_000, 100_000)]


async def test_register_duplicate_username(client, session):
    await register(client)
    resp = await register(client)
    assert resp.status_code == 409
    assert resp.json() == {"code": "USERNAME_TAKEN", "params": {}}
    assert await session.scalar(select(func.count()).select_from(Player)) == 1


async def test_register_validates_input(client):
    assert (await register(client, password="short")).status_code == 422
    assert (await register(client, username="a b")).status_code == 422


async def test_login_success_returns_current_balance(client):
    await register(client)
    resp = await client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
    assert resp.status_code == 200
    assert resp.json()["balance"] == 100_000


async def test_login_rejects_wrong_password_and_unknown_user(client):
    await register(client)
    for creds in (
        {"username": "alice", "password": "wrong password"},
        {"username": "nobody", "password": PASSWORD},
    ):
        resp = await client.post("/api/auth/login", json=creds)
        assert resp.status_code == 401
        assert resp.json()["code"] == "INVALID_CREDENTIALS"


async def test_guest_gets_credits_and_cannot_log_in_again(client):
    resp = await client.post("/api/auth/guest", json={"locale": "sr"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["is_guest"] is True
    assert body["username"].startswith("guest-")
    assert body["balance"] == 100_000

    login = await client.post(
        "/api/auth/login", json={"username": body["username"], "password": PASSWORD}
    )
    assert login.status_code == 401


async def test_guest_without_body_and_unique_names(client):
    a = await client.post("/api/auth/guest")
    b = await client.post("/api/auth/guest")
    assert a.status_code == b.status_code == 201
    assert a.json()["username"] != b.json()["username"]


async def test_current_player_accepts_valid_token_only(client):
    body = (await register(client)).json()

    @app.get("/api/_whoami")
    async def whoami(player: Player = Depends(current_player)):
        return {"username": player.username}

    ok = await client.get("/api/_whoami", headers={"Authorization": f"Bearer {body['token']}"})
    assert ok.json() == {"username": "alice"}

    for headers in ({}, {"Authorization": "Bearer garbage"}):
        resp = await client.get("/api/_whoami", headers=headers)
        assert resp.status_code == 401
        assert resp.json()["code"] == "NOT_AUTHENTICATED"

    expired = create_token(uuid.UUID(body["player_id"]), "dev-only-change-me", -10)
    resp = await client.get("/api/_whoami", headers={"Authorization": f"Bearer {expired}"})
    assert resp.status_code == 401
