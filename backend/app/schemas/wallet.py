from pydantic import BaseModel


class BalanceResponse(BaseModel):
    balance: int  # minor units
