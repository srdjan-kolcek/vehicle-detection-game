import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    jwt_secret: str
    jwt_ttl_s: int
    demo_start_credits: int  # whole credits; the wallet stores minor units (x 100)
    demo_player_password: str


def get_settings() -> Settings:
    return Settings(
        database_url=os.environ["DATABASE_URL"],
        jwt_secret=os.environ.get("JWT_SECRET", "dev-only-change-me"),
        jwt_ttl_s=int(os.environ.get("JWT_TTL_S", "3600")),
        demo_start_credits=int(os.environ.get("DEMO_START_CREDITS", "1000")),
        demo_player_password=os.environ.get("DEMO_PLAYER_PASSWORD", "demo1234"),
    )
