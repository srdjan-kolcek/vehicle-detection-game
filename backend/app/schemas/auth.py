import uuid

from pydantic import BaseModel, Field


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class RegisterRequest(Credentials):
    locale: str = Field(default="en", pattern=r"^(en|sr)$")


class GuestRequest(BaseModel):
    locale: str = Field(default="en", pattern=r"^(en|sr)$")


class SessionResponse(BaseModel):
    token: str
    player_id: uuid.UUID
    username: str
    is_guest: bool
    balance: int  # minor units
