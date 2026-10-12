from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import current_player
from app.db import get_session
from app.engine.bets import place_bet
from app.engine.loop import utc_now
from app.models import Player
from app.schemas.bets import BetRequest, BetResponse
from app.wallet.virtual import VirtualWallet

router = APIRouter(prefix="/rounds", tags=["bets"])


@router.post("/{round_id}/bet", response_model=BetResponse, status_code=201)
async def post_bet(
    round_id: int,
    body: BetRequest,
    response: Response,
    player: Player = Depends(current_player),
    session: AsyncSession = Depends(get_session),
) -> BetResponse:
    """Place the player's one bet in this round. Bets are final. A resubmitted bet_id answers 200."""
    wallet = VirtualWallet(session)
    bet, created = await place_bet(
        session, wallet, player_id=player.id, round_id=round_id, bet_id=body.bet_id,
        choice=body.choice, stake=body.stake, now=utc_now(),
    )
    balance = await wallet.get_balance(player.id)
    await session.commit()
    if not created:
        response.status_code = 200
    return BetResponse(
        bet_id=bet.id, round_id=bet.round_id, choice=bet.choice, stake=bet.stake,
        multiplier=bet.multiplier, status=bet.status, balance=balance,
    )
