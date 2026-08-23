import asyncio
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
import corpus
import embeddings
import runtime
import tools
from index import Index

INDEX = Index()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Try to build at startup, but never refuse to start because a model
    # backend isn't up yet -- the server not-yet-having-a-backend is the
    # common case the Qt side is built to display, not an error (§6.1).
    await _ensure_index()
    yield
    # A backend this server started itself must not outlive it -- an
    # orphaned llama-server holding the GPU is a worse failure mode than the
    # server just not offering /chat until someone restarts it manually.
    await runtime.stop_all_backends()


app = FastAPI(lifespan=lifespan)

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

# Reachable corpus for A2a (docs/SCOPE.md §6.5): whole-corpus context, no
# retrieval infrastructure yet — small enough to paste directly. Each app's
# deeper docs/ subdirectory (e.g. apps/motor_control/docs/RIG_ACCESS.md) is
# deliberately excluded; widen this once retrieval, not stuffing, is in place.
_APPS = ["data_collection", "motor_control", "mlops", "ota", "agent"]

# Now that server/ is its own repository, AGENT_MAESTRO_ROOT has no default
# to fall back on (config.py) -- an empty or wrong value here would silently
# build an empty corpus (Path("") resolves to ".") rather than the intended
# one, and the server would look "up" while actually answering from nothing.
# Fail at startup instead, with the fix spelled out, not a bare traceback.
if not config.AGENT_MAESTRO_ROOT or not (Path(config.AGENT_MAESTRO_ROOT) / "CLAUDE.md").exists():
    raise RuntimeError(
        f"AGENT_MAESTRO_ROOT is not set to a real PdM Maestro checkout "
        f"(got {config.AGENT_MAESTRO_ROOT!r}, expected a directory containing "
        f"CLAUDE.md). Run setup.py, set it in {config.CONFIG_PATH}, or export "
        f"the AGENT_MAESTRO_ROOT environment variable."
    )

CORPUS_PATHS = (
    ["CLAUDE.md"]
    + sorted(f"docs/{p.name}" for p in (Path(config.AGENT_MAESTRO_ROOT) / "docs").glob("*.md"))
    + ["core/README.md"]
    + [f"apps/{app}/README.md" for app in _APPS]
    + sorted(
        f"apps/agent/docs/{p.name}"
        for p in (Path(config.AGENT_MAESTRO_ROOT) / "apps/agent/docs").glob("*.md")
    )
)


CHUNKS = corpus.load_chunks(CORPUS_PATHS)


async def _ensure_index() -> bool:
    """Build the index if it isn't built. Cheap and idempotent once it is."""
    if INDEX.ready:
        return True
    await INDEX.build(CHUNKS)
    return INDEX.ready


def _build_prompt(hits: list[tuple[corpus.Chunk, float]]) -> tuple[str, list[dict]]:
    """The grounded system prompt, and the numbering the answer will refer to.

    Two choices worth stating, both from how small models actually fail:

    * **The model cites `[2]`, not `(docs/STATUS.md § pdm_app_core)`.** A 3B
      model garbles a filename far more readily than a single digit, and the
      server can map the digit back to the real file with no chance of a typo.
      A2a asked for the parenthetical directly; this is a correction, not a
      change of mind about §7 -- the requirement is still that every claim is
      traceable, and this makes it *more* reliable, not less.

    * **Strongest evidence goes last, nearest the question.** Attention is
      weakest in the middle of a context window, and that hurts a small model
      most. So the sources are presented weakest-first: `[5]` is the best
      match, sitting immediately above the question, not buried in the middle.
    """
    ordered = list(reversed(hits))  # weakest first, best adjacent to the question
    sources = []
    blocks = []
    for n, (chunk, score) in enumerate(ordered, start=1):
        sources.append(
            {
                "n": n,
                "path": chunk.path,
                "heading": chunk.heading,
                "citation": chunk.citation,
                "score": round(score, 4),
            }
        )
        blocks.append(f"[{n}] {chunk.citation}\n{chunk.text}")

    prompt = (
        "You are a developer assistant for the PdM Maestro toolchain, a "
        "Qt/QML application that merges several tools for a predictive-"
        "maintenance motor rig.\n\n"
        "Answer only from the numbered sources below. After every factual "
        "claim, write the number of the source it came from in square "
        "brackets, like [2]. Do not write file names -- only the numbers. "
        "If the sources do not cover what was asked, say so plainly instead "
        "of guessing; that is a correct answer, not a failure. When your "
        "answer explains how to use something that lives on a specific tab, "
        "call navigate_to for that tab so the user ends up looking at the "
        "thing you just explained, not just reading about it.\n\n"
        "SOURCES\n\n" + "\n\n".join(blocks)
    )
    return prompt, sources


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _cited_numbers(answer: str, count: int) -> set[int]:
    """Which sources the answer actually pointed at, ignoring invented ones."""
    return {n for n in (int(m) for m in _CITATION_RE.findall(answer)) if 1 <= n <= count}


async def _check_citations(answer: str, hits: list[tuple[corpus.Chunk, float]],
                           cited: set[int]) -> dict:
    """Does the answer resemble the source it claims, more than the ones it doesn't?

    This exists because of an observed failure, not a hypothetical one. Asked
    what breaks if the data is audio, qwen2.5-3b answered correctly *from*
    SCOPE.md §7 and then cited the ESP32 firmware section instead -- the
    wrong-but-adjacent citation that small models are known for. The answer
    was right and the citation was wrong, and nothing in the response said so.

    So: embed the answer, score it against every source that was retrieved,
    and report which one it actually resembles. If that source is not among
    the ones it cited, the citation is unsupported and the caller is told,
    rather than the answer passing for grounded because a digit appeared in
    square brackets somewhere.

    A similarity check is weaker than an entailment model, and it is honest
    to say so: it catches citing the wrong chunk, not a claim that is
    plausible-sounding and absent from every chunk. It costs one embedding
    call against infrastructure that is already loaded, and needs no second
    model on a card that has 1.1 GB left.
    """
    ordered = list(reversed(hits))

    # Asked for the emergency-stop ramp time, the model once answered "2.5"
    # and nothing else. Three characters embed to something close to noise --
    # every support score fell to ~0.45, against ~0.85 for a normal answer --
    # so any verdict computed from them is arithmetic, not evidence. Say the
    # check did not run, which is a third outcome the UI already draws.
    if len(answer.strip()) < config.CITATION_MIN_ANSWER_CHARS:
        return {"checked": False, "supported": None, "best_supported": None,
                "reason": "answer too short to check"}

    try:
        answer_vector = (await embeddings.embed_documents([answer]))[0]
    except embeddings.EmbeddingError:
        # The answer already exists; failing to grade it must not discard it.
        return {"checked": False, "supported": None, "best_supported": None}

    support = {
        n: embeddings.cosine(answer_vector, INDEX.vector_of(chunk))
        for n, (chunk, _) in enumerate(ordered, start=1)
    }
    best = max(support, key=support.get)
    if not cited:
        return {
            "checked": True,
            "supported": False,
            "best_supported": best,
            "margin": None,
            "support": {n: round(v, 4) for n, v in support.items()},
        }

    # Not "is the cited source the single highest scorer" -- that was the
    # first version and it cried wolf. Asked which Qt version the project
    # needs, the model answered correctly in one sentence and cited the
    # section that says so; the check called it a mismatch because a Qt
    # *troubleshooting* section scored 0.821 against the answer's 0.782.
    #
    # A one-sentence answer is the problem: cosine against a 600-character
    # chunk measures topical overlap, and on a short answer topic swamps the
    # much smaller signal of "the fact is in here". So the cited source only
    # has to be *within reach* of the best one, not beat it.
    best_cited = max(support[n] for n in cited)
    margin = support[best] - best_cited
    return {
        "checked": True,
        "supported": margin <= config.CITATION_MARGIN,
        "best_supported": best,
        "margin": round(margin, 4),
        "support": {n: round(v, 4) for n, v in support.items()},
    }

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


class HistoryTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    model: str | None = None
    # Follow-up questions: the client's own transcript, oldest first. Only
    # the question/answer text carries over -- retrieval reruns fresh against
    # the *current* message each turn (SOURCES below is always this turn's
    # top-k, not a running set), so a follow-up gets conversational context
    # ("what did I just ask") without the prompt growing by the full source
    # text of every earlier turn.
    history: list[HistoryTurn] | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


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


class ToolsUnsupportedError(RuntimeError):
    """The backend rejected a request that carried `tools` -- read as "this
    model/runtime combination doesn't do structured tool calls", the thing
    docs/SCOPE.md §6.3 calls a capability probe. There is no separate probe
    request: folding the check into the first real call avoids either
    blocking startup on a backend that isn't up yet (the same reason
    `_ensure_index` never does) or sending the same request twice."""


# Session-scoped, plain module state: one process serves one Qt client, so
# there is no concurrent-session case to guard against here. Amended per
# docs/SCOPE.md §6.3: tools default OFF (a 3B model gets roughly half its
# tool calls right on the design-floor hardware), upgraded by success,
# downgraded permanently for the session once failures pile up.
#
# "Failures" here means the model's tool_call itself didn't parse -- a real
# `qwen2.5-3b-instruct` session, live, hit exactly this the other way:
# picking a wrong-but-valid-looking argument (a real tab, just not the one
# HIGHLIGHT_TARGETS says has that element) three times tripped this counter
# and disabled tools for the rest of the session, on a model that had also
# produced several fully correct calls in the same run. See the comment at
# the increment site in /chat for why only a genuine parse failure counts.
TOOL_PARSE_FAILURE_LIMIT = 3
_tools_state = {"disabled": False, "parse_failures": 0}


def _content_of(message: dict) -> str:
    return message.get("content") or ""


def _tool_call_name_and_args(raw_call: dict) -> tuple[str, dict | None]:
    """`arguments` is a JSON string on the OpenAI-compatible surface
    (llama-server) and already a dict on Ollama's -- normalize both. `None`
    means it didn't parse, which the caller counts as a failure rather than
    guessing at a repair."""
    fn = raw_call.get("function", {})
    name = fn.get("name", "")
    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):
        return name, raw_args
    if isinstance(raw_args, str):
        try:
            return name, json.loads(raw_args)
        except ValueError:
            return name, None
    return name, None


async def _chat_message_llamacpp(messages: list[dict], tool_schema: list[dict] | None) -> dict:
    body = {"messages": messages, "stream": False}
    if tool_schema:
        body["tools"] = tool_schema
    async with httpx.AsyncClient(timeout=config.CHAT_TIMEOUT_SECONDS) as client:
        resp = await client.post(f"{config.LLAMACPP_HOST}/v1/chat/completions", json=body)
        if tool_schema and resp.status_code == 400:
            raise ToolsUnsupportedError(resp.text)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


async def _chat_message_ollama(
    messages: list[dict], model: str | None, tool_schema: list[dict] | None
) -> dict:
    if model is None:
        names = await _fetch_ollama_models()
        if not names:
            raise ValueError("no ollama models available")
        model = names[0]
    body = {"model": model, "messages": messages, "stream": False}
    if tool_schema:
        body["tools"] = tool_schema
    async with httpx.AsyncClient(timeout=config.CHAT_TIMEOUT_SECONDS) as client:
        resp = await client.post(f"{config.OLLAMA_HOST}/api/chat", json=body)
        if tool_schema and resp.status_code == 400:
            raise ToolsUnsupportedError(resp.text)
        resp.raise_for_status()
        return resp.json()["message"]


async def _chat_message(
    messages: list[dict], model: str | None, tool_schema: list[dict] | None
) -> dict:
    """One model call, returning the whole message (content and, maybe,
    tool_calls) rather than just text -- A2a only ever needed the text."""
    if config.BACKEND == "ollama":
        return await _chat_message_ollama(messages, model, tool_schema)
    return await _chat_message_llamacpp(messages, tool_schema)


@app.get("/index/status")
async def index_status():
    return {
        "ready": INDEX.ready,
        "chunks": len(CHUNKS),
        "documents": len(CORPUS_PATHS),
        "error": INDEX.error,
    }


@app.post("/index/rebuild")
async def index_rebuild():
    """Re-embed after the docs change. Cached chunks are not re-embedded."""
    global CHUNKS
    CHUNKS = corpus.load_chunks(CORPUS_PATHS)
    INDEX.ready = False
    if not await _ensure_index():
        return JSONResponse(status_code=503, content={"error": INDEX.error})
    return {"ready": True, "chunks": len(CHUNKS)}


@app.post("/search")
async def search(body: SearchRequest):
    """Retrieval on its own, with no model call -- what the answer is built on.

    Exposed because a citation nobody can check is not a citation: this is how
    you see which sections a question actually pulled, and how a bad answer
    gets diagnosed as bad retrieval versus a bad generation.
    """
    if not await _ensure_index():
        return JSONResponse(status_code=503, content={"error": INDEX.error})
    try:
        hits = await INDEX.search(body.query, body.top_k)
    except embeddings.EmbeddingError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return {
        "results": [
            {
                "path": chunk.path,
                "heading": chunk.heading,
                "citation": chunk.citation,
                "score": round(score, 4),
                "text": chunk.text,
            }
            for chunk, score in hits
        ]
    }


@app.get("/tools/status")
async def tools_status():
    """Whether tool calling is currently on -- diagnosis, same job /search
    does for retrieval: is a bad answer bad tool use, or was there none."""
    return {
        "disabled": _tools_state["disabled"],
        "parse_failures": _tools_state["parse_failures"],
        "failure_limit": TOOL_PARSE_FAILURE_LIMIT,
    }


# ---- runtime bootstrap (runtime.py) --------------------------------------
# The "easy setup" half: a fresh checkout needs no llama.cpp already built
# or installed anywhere, and no shell commands run by hand to start the two
# backend processes this server talks to on :8080/:8081 -- everything this
# session did manually by running `llama-server ...` twice in a terminal is
# available here as an endpoint instead.

@app.get("/runtime/status")
async def runtime_status():
    return {
        "available": runtime.is_available(),
        "binary_path": str(runtime.binary_path()) if runtime.is_available() else None,
    }


@app.post("/runtime/build")
async def runtime_build():
    if runtime.is_available():
        return {"already_available": True}
    if not runtime.start_build():
        return JSONResponse(status_code=409, content={"error": "a build is already in progress"})
    return JSONResponse(status_code=202, content={"started": True})


@app.get("/runtime/build/status")
async def runtime_build_status():
    return dict(runtime._build_state)


@app.get("/runtime/backend/status")
async def runtime_backend_status():
    return {
        "chat": runtime.backend_running("chat"),
        "embed": runtime.backend_running("embed"),
    }


class BackendStartRequest(BaseModel):
    chat_model: str  # an -hf repo:quant id, same format as the CATALOG entries
    embed_model: str = "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"


@app.post("/runtime/backend/start")
async def runtime_backend_start(body: BackendStartRequest):
    if not runtime.is_available():
        return JSONResponse(status_code=503, content={"error": "no llama-server binary -- POST /runtime/build first"})
    try:
        # Chat first, embed second: if only one GPU-resident model fits
        # comfortably alongside headroom, the chat model is the one on the
        # latency path every single question needs (§6.4) -- the embed
        # process below already forces itself onto the CPU regardless.
        await runtime.start_chat_backend(body.chat_model, port=8080)
        await runtime.start_embed_backend(body.embed_model, port=8081)
    except Exception as exc:  # missing binary, bad model id, port in use
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return {"chat": runtime.backend_running("chat"), "embed": runtime.backend_running("embed")}


@app.post("/runtime/backend/stop")
async def runtime_backend_stop():
    await runtime.stop_all_backends()
    return {"chat": runtime.backend_running("chat"), "embed": runtime.backend_running("embed")}


@app.post("/chat")
async def chat(body: ChatRequest):
    if not await _ensure_index():
        return JSONResponse(
            status_code=503,
            content={"error": f"retrieval unavailable: {INDEX.error}"},
        )
    try:
        hits = await INDEX.search(body.message)
    except embeddings.EmbeddingError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    system_prompt, sources = _build_prompt(hits)
    messages = [{"role": "system", "content": system_prompt}]
    if body.history:
        messages.extend({"role": turn.role, "content": turn.content} for turn in body.history)
    messages.append({"role": "user", "content": body.message})

    # The deterministic retrieve-then-answer path above always runs, tools
    # or not (docs/SCOPE.md §6.3 amended). What follows is strictly additive:
    # offer the model a chance to call one, and fall back to its plain answer
    # on any sign the offer wasn't a good idea.
    offer_tools = not _tools_state["disabled"]
    tool_log: list[dict] = []
    try:
        try:
            msg = await _chat_message(messages, body.model, tools.TOOL_SCHEMAS if offer_tools else None)
        except ToolsUnsupportedError:
            _tools_state["disabled"] = True
            offer_tools = False
            msg = await _chat_message(messages, body.model, None)

        raw_calls = (msg.get("tool_calls") or []) if offer_tools else []
        if raw_calls:
            # Bounded to the first call, one round. A full ReAct loop is next
            # to build only once a single round is proven not to be enough --
            # docs/SCOPE.md §4 deferred exactly this, for the same reason.
            raw_call = raw_calls[0]
            name, arguments = _tool_call_name_and_args(raw_call)
            if arguments is None:
                _tools_state["parse_failures"] += 1
                tool_log.append(
                    {"name": name, "arguments": None,
                     "result": {"error": "arguments did not parse as JSON"}}
                )
                answer = _content_of(msg)
            else:
                result = await tools.dispatch(
                    name, arguments, index=INDEX, corpus_paths=CORPUS_PATHS
                )
                # A well-formed call with a wrong argument value -- an
                # unknown tab, a section that isn't on that tab -- is not
                # counted here. Live testing found this the hard way: three
                # such rejections (each a real, working validation doing
                # exactly its job -- see tools.py's HIGHLIGHT_TARGETS check)
                # tripped the counter and disabled tools for the rest of the
                # session, on a model that had *also* produced several fully
                # correct calls in the same session. That is the model
                # exploring a real but small option space, not evidence it
                # can't do structured calling -- the thing this counter is
                # actually meant to catch, and `arguments is None` above
                # already catches it. Counting semantic misses here would
                # make the fallback fire *because* the model is doing
                # exactly what §6.3 asks of it: propose, get told no by a
                # real allowlist, and the caller still gets a clean error to
                # recover from -- disabling tools over that punishes the
                # validation working, not a validation gap.
                tool_log.append({"name": name, "arguments": arguments, "result": result})
                followup = messages + [
                    {"role": "assistant", "content": msg.get("content") or "",
                     "tool_calls": msg.get("tool_calls")},
                    {"role": "tool", "tool_call_id": raw_call.get("id", name),
                     "content": json.dumps(result)},
                ]
                final_msg = await _chat_message(followup, body.model, None)
                answer = _content_of(final_msg)
            if _tools_state["parse_failures"] >= TOOL_PARSE_FAILURE_LIMIT:
                _tools_state["disabled"] = True
        else:
            answer = _content_of(msg)
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": f"{config.BACKEND} at {_backend_host()} timed out after {config.CHAT_TIMEOUT_SECONDS}s"},
        )
    except (httpx.HTTPError, ValueError) as exc:
        return JSONResponse(
            status_code=503,
            content={"error": f"could not reach {config.BACKEND} at {_backend_host()}: {exc}"},
        )
    except Exception as exc:  # malformed model response, etc. — never crash the server
        return JSONResponse(status_code=500, content={"error": str(exc)})

    cited = _cited_numbers(answer, len(sources))
    check = await _check_citations(answer, hits, cited)
    return {
        "answer": answer,
        # Whether the answer resembles what it says it came from -- see
        # _check_citations. `grounded` alone means only that some number was
        # printed; this is the part that says whether it was the right one.
        "citation_check": check,
        # Every source the answer was built from, each flagged with whether
        # the model actually pointed at it. An uncited answer is visible as
        # such rather than quietly passing for a grounded one -- §7 makes
        # citation a requirement, so the absence of one is information.
        "sources": [{**s, "cited": s["n"] in cited} for s in sources],
        "grounded": bool(cited),
        # A2b: which tool, if any, the model called this turn, and what it
        # got back. Empty on the common path. The Qt client ignores unknown
        # fields today (this is how `sources` landed too, in A2a), so this is
        # forward-looking: the UI to show/act on it is not built yet.
        "tool_calls": tool_log,
    }
