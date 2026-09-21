import time
import uuid

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app.core.errors import AppError

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def create_token(player_id: uuid.UUID, secret: str, ttl_s: int) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(player_id), "iat": now, "exp": now + ttl_s}, secret, "HS256")


def decode_token(token: str, secret: str) -> uuid.UUID:
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp", "sub"]})
        return uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, ValueError):
        raise AppError("NOT_AUTHENTICATED") from None
