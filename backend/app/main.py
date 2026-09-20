from fastapi import FastAPI

app = FastAPI(title="Vehicle Detection Game")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/hello")
def hello() -> dict:
    return {"message": "Hello from the backend"}
