"""
Turning text into vectors, on whichever backend is configured.

Two things here are easy to get wrong and fail *silently* -- no exception, no
error, just worse answers -- so both are done in one place:

  1. nomic-embed-text is asymmetric. A document embedded as though it were a
     query lands in a different part of the space and stops matching. The
     prefixes are in config.py; nothing else may embed without going through
     these two functions.
  2. Cosine similarity assumes unit vectors. nomic's are already normalized,
     but llama-server does not consistently apply --embd-normalize, so we
     normalize on this side regardless rather than trusting the backend.
"""

import math

import httpx

import config


class EmbeddingError(RuntimeError):
    """The embedding backend is unreachable or answered with nonsense."""


def _embed_host() -> str:
    return config.OLLAMA_HOST if config.BACKEND == "ollama" else config.LLAMACPP_EMBED_HOST


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        raise EmbeddingError("embedding backend returned a zero vector")
    return [x / norm for x in vector]


async def _embed_ollama(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=config.EMBED_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            f"{config.OLLAMA_HOST}/api/embed",
            json={"model": config.OLLAMA_EMBED_MODEL, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


async def _embed_llamacpp(texts: list[str]) -> list[list[float]]:
    # The OpenAI-compatible surface of a *second* llama-server, started with
    # --embeddings and the nomic GGUF. See config.LLAMACPP_EMBED_HOST for why
    # it cannot be the same process as the chat model.
    async with httpx.AsyncClient(timeout=config.EMBED_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            f"{config.LLAMACPP_EMBED_HOST}/v1/embeddings",
            json={"input": texts},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # Order is not promised by the schema; index is. Sort rather than assume.
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]


async def _embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        raw = (
            await _embed_ollama(texts)
            if config.BACKEND == "ollama"
            else await _embed_llamacpp(texts)
        )
    except (httpx.HTTPError, KeyError, TypeError) as exc:
        raise EmbeddingError(f"could not embed via {config.BACKEND} at {_embed_host()}: {exc}")
    if len(raw) != len(texts):
        raise EmbeddingError(f"asked for {len(texts)} embeddings, got {len(raw)}")
    return [_normalize(v) for v in raw]


async def embed_documents(texts: list[str]) -> list[list[float]]:
    return await _embed([config.EMBED_DOCUMENT_PREFIX + t for t in texts])


async def embed_query(text: str) -> list[float]:
    return (await _embed([config.EMBED_QUERY_PREFIX + text]))[0]


def cosine(a: list[float], b: list[float]) -> float:
    """Both sides are unit vectors by the time they get here, so: a dot b.

    No numpy. At 133 chunks x 768 dimensions this is ~100k multiply-adds per
    question -- a few milliseconds, against a model call measured in seconds.
    A dependency would buy nothing until the corpus is a couple of orders of
    magnitude larger, and this server has to install on machines weaker than
    the one it was written on (§6.4).
    """
    return sum(x * y for x, y in zip(a, b))
