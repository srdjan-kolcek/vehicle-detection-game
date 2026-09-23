from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import auth, wallet
from app.core.errors import AppError

app = FastAPI(title="Vehicle Detection Game")

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
