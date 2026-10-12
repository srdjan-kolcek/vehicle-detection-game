import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import auth, wallet
from app.core.config import get_settings
from app.core.errors import AppError
from app.db import get_engine
from app.engine import RoundEngine

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "info").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = RoundEngine(get_engine(), settings.pipeline_signing_key) if settings.round_engine_enabled else None
    if engine is not None:
        await engine.start()
    app.state.round_engine = engine
    yield
    if engine is not None:
        await engine.stop()


app = FastAPI(title="Vehicle Detection Game", lifespan=lifespan)

_STATUS = {
    "NOT_AUTHENTICATED": 401,
    "INVALID_CREDENTIALS": 401,
    "USERNAME_TAKEN": 409,
    "REHYDRATE_COOLDOWN": 429,
}


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=_STATUS.get(exc.code, 400), content={"code": exc.code, "params": exc.params}
    )


app.include_router(auth.router, prefix="/api")
app.include_router(wallet.router, prefix="/api")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/hello")
def hello() -> dict:
    return {"message": "Hello from the backend"}
