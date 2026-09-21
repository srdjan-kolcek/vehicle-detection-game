import uuid
from typing import Protocol

# Integer minor units (credits x 100).
Money = int


class WalletAdapter(Protocol):
    """What the game needs from a wallet. An operator replaces the implementation.

    All calls are idempotent by bet_id and run inside the caller's transaction.
    """

    async def get_balance(self, player_id: uuid.UUID) -> Money: ...

    async def place_bet(
        self, player_id: uuid.UUID, bet_id: uuid.UUID, round_id: int, amount: Money
    ) -> None: ...

    async def settle(self, bet_id: uuid.UUID, payout: Money) -> None: ...

    async def rollback(self, bet_id: uuid.UUID) -> None: ...
