from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import current_player
from app.core.config import Settings, get_settings
from app.db import get_session
from app.models import Player
from app.schemas.wallet import BalanceResponse
from app.wallet.virtual import VirtualWallet

router = APIRouter(prefix="/wallet", tags=["wallet"])


@router.post("/rehydrate", response_model=BalanceResponse)
async def rehydrate(
    player: Player = Depends(current_player),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> BalanceResponse:
    """Demo top-up to DEMO_START_CREDITS worth of minor units, gated by REHYDRATE_COOLDOWN_S."""
    balance = await VirtualWallet(session).rehydrate(
        player.id, settings.demo_start_credits * 100, settings.rehydrate_cooldown_s
    )
    await session.commit()
    return BalanceResponse(balance=balance)
