import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import config

app = FastAPI()


async def _fetch_ollama_models() -> list[str]:
    async with httpx.AsyncClient(timeout=config.BACKEND_TIMEOUT_SECONDS) as client:
        resp = await client.get(f"{config.OLLAMA_HOST}/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]


async def _fetch_llamacpp_models() -> list[str]:
    # OpenAI-compatible surface: {"object": "list", "data": [{"id": ..., ...}]}.
    # llama-server has exactly one model loaded, so this is a one-entry list;
    # switching models means pointing LLAMACPP_HOST at a different process.
    async with httpx.AsyncClient(timeout=config.BACKEND_TIMEOUT_SECONDS) as client:
        resp = await client.get(f"{config.LLAMACPP_HOST}/v1/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]


def _backend_host() -> str:
    return config.OLLAMA_HOST if config.BACKEND == "ollama" else config.LLAMACPP_HOST


async def _fetch_models() -> list[str]:
    if config.BACKEND == "ollama":
        return await _fetch_ollama_models()
    return await _fetch_llamacpp_models()


@app.get("/health")
async def health():
    try:
        await _fetch_models()
        connected = True
    except (httpx.HTTPError, ValueError):
        connected = False
    return {
        "ok": True,
        "backend": config.BACKEND,
        "connected": connected,
        "host": _backend_host(),
    }


@app.get("/models")
async def models():
    try:
        names = await _fetch_models()
    except (httpx.HTTPError, ValueError) as exc:
        return JSONResponse(
            status_code=503,
            content={"error": f"could not reach {config.BACKEND} at {_backend_host()}: {exc}"},
        )
    return {"models": names}
