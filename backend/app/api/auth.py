import secrets
import uuid

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.security import create_token, decode_token, hash_password, verify_password
from app.db import get_session
from app.models import Player, WalletAccount
from app.schemas.auth import Credentials, GuestRequest, RegisterRequest, SessionResponse
from app.wallet.virtual import VirtualWallet

router = APIRouter(prefix="/auth", tags=["auth"])
_bearer = HTTPBearer(auto_error=False)

# Verified against when the username is unknown, so a miss costs as much as a wrong password.
_DUMMY_HASH = hash_password("not-a-real-password")


async def current_player(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Player:
    if creds is None:
        raise AppError("NOT_AUTHENTICATED")
    player = await session.get(Player, decode_token(creds.credentials, settings.jwt_secret))
    if player is None:
        raise AppError("NOT_AUTHENTICATED")
    return player


async def _open_account(session: AsyncSession, player: Player, settings: Settings) -> int:
    """Create the wallet and credit the starting balance through the ledger. Returns the balance."""
    session.add(player)
    await session.flush()
    session.add(WalletAccount(player_id=player.id, balance=0))
    await session.flush()
    return await VirtualWallet(session).topup(player.id, settings.demo_start_credits * 100)


def _session_response(
    player: Player, balance: int, settings: Settings
) -> SessionResponse:
    return SessionResponse(
        token=create_token(player.id, settings.jwt_secret, settings.jwt_ttl_s),
        player_id=player.id,
        username=player.username,
        is_guest=player.password_hash is None,
        balance=balance,
    )


@router.post("/register", response_model=SessionResponse, status_code=201)
async def register(
    body: RegisterRequest,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> SessionResponse:
    player = Player(
        id=uuid.uuid4(),
        username=body.username,
        password_hash=hash_password(body.password),
        locale=body.locale,
    )
    try:
        balance = await _open_account(session, player, settings)
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise AppError("USERNAME_TAKEN") from None
    return _session_response(player, balance, settings)


@router.post("/login", response_model=SessionResponse)
async def login(
    body: Credentials,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> SessionResponse:
    player = await session.scalar(select(Player).where(Player.username == body.username))
    stored = player.password_hash if player else None
    ok = verify_password(stored or _DUMMY_HASH, body.password)
    if player is None or stored is None or not ok:
        raise AppError("INVALID_CREDENTIALS")
    balance = await VirtualWallet(session).get_balance(player.id)
    return _session_response(player, balance, settings)


@router.post("/guest", response_model=SessionResponse, status_code=201)
async def guest(
    body: GuestRequest | None = None,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> SessionResponse:
    """A throwaway player with starting credits; it has no password and cannot log in again."""
    player = Player(
        id=uuid.uuid4(),
        username=f"guest-{secrets.token_hex(4)}",
        password_hash=None,
        locale=body.locale if body else "en",
    )
    balance = await _open_account(session, player, settings)
    await session.commit()
    return _session_response(player, balance, settings)
