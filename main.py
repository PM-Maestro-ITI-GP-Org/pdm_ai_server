import asyncio
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config

app = FastAPI()

# Curated, not fetched from HF's search API — four known-good targets,
# spanning the hardware tiers in docs/SCOPE.md §6.4. Filenames and
# size_bytes are the real values for each repo's Q4_K_M file, resolved by
# hand against the HF API; keep them in sync if a repo re-quantizes.
CATALOG = [
    {
        "id": "qwen2.5-1.5b-instruct-q4",
        "label": "Qwen2.5 1.5B Instruct (Q4_K_M)",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_bytes": 1117320736,
    },
    {
        "id": "qwen2.5-3b-instruct-q4",
        "label": "Qwen2.5 3B Instruct (Q4_K_M)",
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_bytes": 2104932768,
    },
    {
        "id": "qwen2.5-7b-instruct-q4",
        "label": "Qwen2.5 7B Instruct (Q4_K_M)",
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_bytes": 4683074240,
    },
    {
        "id": "llama-3.2-3b-instruct-q4",
        "label": "Llama 3.2 3B Instruct (Q4_K_M)",
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_bytes": 2019377696,
    },
]

DOWNLOAD_CHUNK_BYTES = 1 << 20

# One download at a time, no queue (§6 keeps this phase deliberately small).
# Plain module-level state is safe here: every mutation happens on the
# asyncio event loop with no `await` between check and set.
_download_state = {
    "active": False,
    "id": None,
    "bytes_downloaded": 0,
    "bytes_total": 0,
    "done": False,
    "error": None,
}


class DownloadRequest(BaseModel):
    id: str


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


def _models_dir() -> Path:
    path = Path(config.AGENT_MODELS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _catalog_entry(model_id: str) -> dict | None:
    return next((m for m in CATALOG if m["id"] == model_id), None)


def _is_installed(entry: dict) -> bool:
    path = _models_dir() / entry["filename"]
    return path.exists() and path.stat().st_size == entry["size_bytes"]


@app.get("/catalog")
async def catalog():
    return {"models": [{**entry, "installed": _is_installed(entry)} for entry in CATALOG]}


async def _run_download(entry: dict) -> None:
    dest_dir = _models_dir()
    final_path = dest_dir / entry["filename"]
    partial_path = dest_dir / f"{entry['filename']}.partial"
    url = f"https://huggingface.co/{entry['repo']}/resolve/main/{entry['filename']}"
    try:
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(partial_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(DOWNLOAD_CHUNK_BYTES):
                        f.write(chunk)
                        _download_state["bytes_downloaded"] += len(chunk)
        if _download_state["bytes_downloaded"] != entry["size_bytes"]:
            raise ValueError(
                f"size mismatch: got {_download_state['bytes_downloaded']}, "
                f"expected {entry['size_bytes']}"
            )
        partial_path.rename(final_path)
        _download_state["done"] = True
    except Exception as exc:  # network error, disk full, size mismatch — never fail silently
        _download_state["error"] = str(exc)
        partial_path.unlink(missing_ok=True)
    finally:
        _download_state["active"] = False


@app.post("/download")
async def download(body: DownloadRequest):
    entry = _catalog_entry(body.id)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": "unknown model id"})
    if _is_installed(entry):
        return {"already_installed": True}
    if _download_state["active"]:
        return JSONResponse(status_code=409, content={"error": "a download is already in progress"})
    _download_state.update(
        active=True,
        id=entry["id"],
        bytes_downloaded=0,
        bytes_total=entry["size_bytes"],
        done=False,
        error=None,
    )
    asyncio.create_task(_run_download(entry))
    return JSONResponse(status_code=202, content={"started": True})


@app.get("/download/status")
async def download_status():
    total = _download_state["bytes_total"]
    downloaded = _download_state["bytes_downloaded"]
    return {
        "active": _download_state["active"],
        "id": _download_state["id"],
        "bytes_downloaded": downloaded,
        "bytes_total": total,
        "percent": round(downloaded / total * 100, 1) if total else 0,
        "done": _download_state["done"],
        "error": _download_state["error"],
    }
