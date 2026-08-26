# pdm_ai_server

Standalone local AI inference server for the PdM Maestro toolchain — the
backend behind the AI Agent tab ([pdm_ai_agent_gui](https://github.com/PM-Maestro-ITI-GP-Org/pdm_ai_agent_gui)).
Runs outside Qt, talks to a local model backend, and is reached over HTTP by
the Qt client. Phase A2b is complete on both halves: a retrieval-grounded
`/chat` with citation checking, and the tool half — the model can search,
read and navigate the running app. On top of that sits the runtime bootstrap
(`runtime.py`): a fresh clone can build `llama-server`, download models, and
start both backend processes from inside the server itself, no manual
terminal work left.

> `docs/SCOPE.md` references below point at `apps/agent/docs/SCOPE.md` in the
> GUI repo, where the phase plan and hardware measurements live.

## Install

The one-command path, from this repo's root:

```bash
python3 setup.py
```

It writes the config file, creates the venv, optionally builds llama-server
(multi-minute; skipped by default), and pre-configures the Qt side's server
URL and its "Start local AI" command. Rerunnable anytime; every prompt takes
its default from Enter alone.

By hand instead (same effect, from this repo's root):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port ${AGENT_SERVER_PORT:-8420}
```

or press **Start local AI** in the AI Agent tab of the Qt app — the client
launches exactly this, detached, from the command setup.py wrote into its
settings (`agent/serverStartCommand`). Starting uvicorn is the only thing Qt
ever owns; everything after that first exec stays server-side (§6.1).

## Config

A TOML file at `~/.config/pdm_agent/config.toml` (override the location with
`AGENT_CONFIG_PATH`) holds the same keys as the environment; priority is
**environment variable > config file > built-in default**, so a systemd unit
or docker file can override the file underneath without editing it. setup.py
writes the file; editing by hand works too. All of it is optional:

| Variable | Default | |
|---|---|---|
| `AGENT_MAESTRO_ROOT` | *(none)* | the PdM Maestro checkout the corpus is read from — must contain `CLAUDE.md`; unset or wrong stops the server at startup with instructions, rather than silently answering from an empty corpus |
| `AGENT_BACKEND` | `llamacpp` | `llamacpp` or `ollama` |
| `LLAMACPP_HOST` | `http://localhost:8080` | chat model, used when `AGENT_BACKEND=llamacpp` |
| `LLAMACPP_EMBED_HOST` | `http://localhost:8081` | embedding model, used when `AGENT_BACKEND=llamacpp` — a *second* `llama-server` process, see "Running the backends" below |
| `OLLAMA_HOST` | `http://localhost:11434` | chat and embeddings both, used when `AGENT_BACKEND=ollama` |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | which pulled Ollama model embeds — independent of which one chats |
| `AGENT_TOP_K` | `5` | sections retrieved per question, see "Retrieval" below |
| `AGENT_INDEX_PATH` | `~/.cache/pdm_agent/index.json` | cached chunk vectors, keyed by content hash, not file mtime |
| `AGENT_SERVER_PORT` | `8420` | |
| `AGENT_MODELS_DIR` | `~/.cache/pdm_agent/models` | downloaded `.gguf` files land here, never in the repo |
| `AGENT_RUNTIME_DIR` | `~/.cache/pdm_agent/runtime` | where the self-built `llama-server` lands (`bin/`) and llama.cpp is cloned (`src/`), see "Runtime bootstrap" below |

Point the relevant host at a second laptop to use it as the model server —
no code change (`docs/SCOPE.md` §3.3, §6.4).

**Why two backends:** `llama-server` loads exactly one model per process —
there's no multi-model registry to list, so under `llamacpp`, "which model"
means "which server," and switching models means pointing `LLAMACPP_HOST` at
a different running `llama-server`. Ollama can list and switch between many
pulled models from one process. Both are supported because the models are
meant to be genuinely interchangeable across machines, not tied to whichever
runtime happens to be installed on this one (`docs/SCOPE.md` §6.4).

Retrieval (see below) needs an embedding model as well as a chat model, and
the two backends differ here in a way that has to show up in configuration
rather than being papered over. Ollama holds several models in one daemon,
so `OLLAMA_HOST` serves both and `OLLAMA_EMBED_MODEL` just names which pulled
model handles embedding calls — the concrete advantage `docs/SCOPE.md` §6.4
claimed for it, now confirmed. `llama-server` still loads exactly one model
per process, so under `llamacpp` the embedding model needs a *second* server
on a *second* port: `LLAMACPP_EMBED_HOST`. llama.cpp's router mode and
llama-swap were both considered for this and rejected — each keeps only one
model resident and pays a 3–10 s reload to switch models, and a switch would
land on every single question, since every question embeds before it chats.

## Running the backends

Under `llamacpp` (the default), two `llama-server` processes run side by
side, one per model. The server can start and stop both itself —
`POST /runtime/backend/start` with an `-hf` model id (see "Runtime bootstrap"
below) — but the manual equivalent is what it runs underneath:

```bash
# chat model, port 8080 — --jinja enables the tool-calling grammar
llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf --port 8080 --jinja

# embedding model, port 8081 — a second process, started with --embeddings.
# -ub/-b raised from the 512 default: the server batches every string in one
# /v1/embeddings call together, and at this corpus's ~140 chunks the combined
# token count exceeds 512 -- found running this against a real corpus for the
# first time, 2026-08-23 (docs/SCOPE.md, A2b's "run against a real backend").
# Below 2048 the call 500s with a clear message rather than crashing, on a
# current build; an older prebuilt binary crashed outright on the same input
# regardless of this setting -- if embeddings crash unpredictably on a full
# corpus batch and raising -ub doesn't help, suspect the binary, not the input.
llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --port 8081 --embeddings \
    --pooling mean -ub 2048 -b 2048
```

Under `ollama`, one daemon serves both, so it's just two pulls:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

`OLLAMA_HOST` already points at that daemon for both models; there is no
`--jinja`-equivalent flag to set.

## Retrieval

A2a pasted the whole reachable corpus into the system prompt. That corpus is
98,717 bytes across 11 documents — roughly 25,000 tokens — and it never fit:
`llama-server`/Ollama realistically get `n_ctx` 2048–4096 on the 4 GB target
hardware. Retrieval exists because of that measured ceiling, not as an
optimization chosen for its own sake.

`corpus.py` splits the corpus at markdown headings — the natural unit of
citation, since `docs/SCOPE.md` §7 makes citing a hard requirement and a
chunk spanning two sections would give the model nothing honest to cite.
Fenced code blocks are never split, even on a line inside them that merely
looks like a heading. The result is 133 chunks, a median of 618 characters
and a max of 1996 (longer sections are windowed further — see
`MAX_CHUNK_CHARS` in `corpus.py`). A top-`k` of 5 (`AGENT_TOP_K`) produces a
roughly 1,970-token prompt, comfortably inside even the 2048-token floor.

`embeddings.py` turns text into vectors with `nomic-embed-text` v1.5, 768
dimensions. It is asymmetric — documents are embedded with a
`"search_document: "` prefix, queries with `"search_query: "` — and getting
that wrong degrades results silently: no exception, just worse answers. Both
prefixes live in one place (`config.py`), and nothing else in the server is
allowed to embed text directly.

`index.py` holds every chunk's vector and searches with a plain Python loop
over unit vectors — `sum(x * y for x, y in zip(a, b))`, no numpy, no vector
database. At 133 chunks × 768 dimensions that's on the order of 100k
multiply-adds: a few milliseconds, against a model call measured in seconds.
The result is cached at `AGENT_INDEX_PATH`, keyed by a sha256 of each
chunk's own content, deliberately not by file mtime — a git checkout
rewrites mtimes without changing a word. Editing one section of one document
re-embeds that one chunk, not all 133.

The plumbing is exercised in tests with a stand-in embedder (`test_tools.py`,
`test_runtime.py`, plain asserts, no framework), and the real
`llama-server`/Ollama path has been run live on this project's actual floor
hardware (GTX 1650) during the A2b session — including the embedding-batch
crash documented above, which only a real backend could have taught us.

## Runtime bootstrap

Everything in this section exists so a fresh machine needs nothing
pre-installed beyond git, cmake, and a C/C++ toolchain — the same tools a
Qt/C++ developer on this project already has.

`GET /runtime/status` → whether a `llama-server` binary is available (either
built into `AGENT_RUNTIME_DIR/bin` or already on `PATH`):

```json
{"available": false, "binary_path": null}
```

`POST /runtime/build` clones llama.cpp shallowly and builds the
`llama-server` target with CUDA enabled whenever `nvcc` is present
(`-DCMAKE_CUDA_ARCHITECTURES=native` — detected from whatever GPU the build
machine actually has, not a guess baked into a prebuilt asset name). It
returns `202 {"started": true}` immediately; progress comes from

`GET /runtime/build/status`:

```json
{"active": true, "stage": "building", "log_tail": ["..."], "done": false, "error": null}
```

A prebuilt download was considered first and rejected: llama.cpp's current
nightly release scheme carries no stable usable asset, and a build-time
bundle would either compile per-GPU-architecture or ship CPU-only. Building
from source is what targets every card from the GTX 1650 floor to an RTX
4070+ with one code path.

Once a binary exists, `POST /runtime/backend/start` starts the two backend
processes themselves:

```json
{"chat_model": "Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M"}
```

→ `{"chat": true, "embed": true}` — chat on :8080, embeddings on :8081 (the
embedding process pins itself to `-ngl 0`: both models GPU-resident at once
crashed both processes on the 4 GB card). `GET /runtime/backend/status`
reports liveness; `POST /runtime/backend/stop` tears both down. Backends this
server started never outlive it — the lifespan handler stops them on exit,
because an orphaned llama-server holding the GPU is the worst way for a GUI
close to end.

## Endpoints

`GET /health` → `200`, always, even if the backend is down:

```json
{"ok": true, "backend": "llamacpp", "connected": true, "host": "http://localhost:8080"}
```

`connected` is a live check against the backend, not a hard-coded value.

`GET /models` → `200` with the available model name(s) — one, under
`llamacpp`; however many are pulled, under `ollama`:

```json
{"models": ["qwen2.5-3b-instruct-q4"]}
```

or `503` if the backend is unreachable:

```json
{"error": "could not reach llamacpp at http://localhost:8080: ..."}
```

`GET /catalog` → a curated, hard-coded list of downloadable models (not a
live Hugging Face search), each flagged with whether it's already on disk:

```json
{"models": [
  {"id": "qwen2.5-3b-instruct-q4", "label": "Qwen2.5 3B Instruct (Q4_K_M)",
   "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF", "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
   "size_bytes": 2104932768, "installed": false}
]}
```

`POST /download` with body `{"id": "qwen2.5-3b-instruct-q4"}` starts a
background download of that catalog entry into `AGENT_MODELS_DIR`, streamed
in chunks to a `.partial` file and renamed only once complete and
size-verified:

- `202 {"started": true}` — download kicked off
- `200 {"already_installed": true}` — file already present with the right size, no-op
- `404 {"error": "unknown model id"}`
- `409 {"error": "a download is already in progress"}` — one at a time, no queue

`GET /download/status` — poll while a download is active:

```json
{"active": true, "id": "qwen2.5-3b-instruct-q4", "bytes_downloaded": 104857600,
 "bytes_total": 2104932768, "percent": 5.1, "done": false, "error": null}
```

On failure (network error, disk full, size mismatch) `error` holds a short
message, `active` and `done` are both `false`, and the partial file is
removed — never a stuck "downloading" that isn't.

`GET /index/status` → whether the retrieval index is built and searchable:

```json
{"ready": true, "chunks": 133, "documents": 11, "error": null}
```

`chunks` and `documents` describe the corpus as loaded, regardless of
whether embedding has finished; `error` carries the reason when `ready` is
`false` — typically that the embedding backend isn't reachable yet.

`POST /index/rebuild` — re-reads the corpus from disk and embeds whatever
changed since the index was last built; chunks already in the cache (keyed
by content hash, see "Retrieval" above) are not re-embedded:

```json
{"ready": true, "chunks": 133}
```

or `503 {"error": "..."}` if the embedding backend can't be reached.

`POST /search` with body `{"query": "how does the queue resume?", "top_k": 3}`
(`top_k` optional, defaults to `AGENT_TOP_K`) → the retrieved sections
themselves, with no model call involved:

```json
{"results": [
  {"path": "docs/STATUS.md",
   "heading": "Bugs found and fixed > 5. Queue resume drops the pause flag",
   "citation": "docs/STATUS.md § Bugs found and fixed > 5. Queue resume drops the pause flag",
   "score": 0.7421, "text": "..."}
]}
```

or `503 {"error": "..."}` if the index isn't ready. It exists on its own
because a citation nobody can check is not a citation: this is how a bad
`/chat` answer gets diagnosed as bad retrieval (the wrong sections came back)
versus bad generation (the right sections came back and the model still got
it wrong).

`POST /chat` with body `{"message": "what tabs does PdM Maestro have?"}` →
an answer grounded in the top `AGENT_TOP_K` retrieved sections, not the whole
corpus (`docs/SCOPE.md` §6.5, §7):

```json
{
  "answer": "PdM Maestro has five tabs: ... [2] ...",
  "sources": [
    {"n": 1, "path": "docs/STATUS.md", "heading": "...",
     "citation": "docs/STATUS.md § ...", "score": 0.61, "cited": false},
    {"n": 2, "path": "CLAUDE.md", "heading": "",
     "citation": "CLAUDE.md", "score": 0.78, "cited": true}
  ],
  "grounded": true
}
```

Every source retrieved for the question comes back, each flagged with
whether the model actually pointed at it. `grounded` is `false` when the
answer cited none of them — visible information about a bad answer, rather
than a silently-passing "grounded" response.

**A2b, tool half: the response also carries `tool_calls`**, empty on the
common path. Before answering, the model is offered four tools —
`search_docs`, `read_file`, `list_repo`, `navigate_to`, defined in
`tools.py` — and may call at most one per question (a bounded single round,
not a ReAct loop; see `docs/SCOPE.md`'s A2b update for why one round is where
this stops for now). A populated entry looks like:

```json
"tool_calls": [
  {"name": "navigate_to", "arguments": {"tab": "motor_control"},
   "result": {"navigate": {"tab": "motor_control", "section": ""}}}
]
```

`navigate_to` validates the tab against the five real ids and hands back an
instruction — it does not touch `MessageBus` itself, which lives in the Qt
process, not here. The Qt client does not act on this field yet; today it is
forward-looking the same way `sources` was in the A2a→A2b transition — an
unknown field an older client just ignores.

**Tool calling defaults off and turns on per session, not globally.** There
is no separate startup probe: offering `tools` on the first real `/chat` call
*is* the probe, folded in rather than duplicated — a backend that 400s on the
field is read as "doesn't support this," and tools switch off for the rest of
the session. They also switch off once three tool calls in a row fail to
parse or fail to execute (`GET /tools/status` reports the running count). A
weak model on the design-floor hardware is expected to hit this — Qwen2.5-3B
gets roughly half its tool calls right on published benchmarks — so the
citation-checked retrieval answer above is what most questions actually get;
the tools are additive, never required for an answer to exist.

`GET /tools/status`:

```json
{"disabled": false, "parse_failures": 0, "failure_limit": 3}
```

**Citation format changed from A2a.** A2a asked the model to write the
filename inline, e.g. `(CLAUDE.md)`. A2b numbers the sources instead and
asks the model to write only the digit, e.g. `[2]`; the server maps that
number back to the real citation (`docs/STATUS.md § pdm_app_core`, for
instance). This is a correction, not a retreat from `docs/SCOPE.md` §7's
citation requirement — a 3B model garbles a filename far more readily than a
single digit, and a number-to-source mapping done by the server cannot typo.
It makes the requirement more reliable, not weaker.

**The strongest-scoring source is presented last**, immediately above the
question, rather than first. Attention is weakest in the middle of a context
window, and that hurts small models the most, so sources are ordered
weakest-first — at `top_k=5`, `[5]` is the best match, sitting right next to
what's being asked.

- `503 {"error": "..."}` — retrieval unavailable (index not built, or the
  embedding backend is unreachable) or the chat backend is unreachable
- `504 {"error": "..."}` — model call timed out (120s budget; local inference
  can be slow, especially on first load)
- `500 {"error": "..."}` — anything else unexpected, never a hang or crash

Under `ollama`, an optional `"model"` field picks which pulled model answers;
it defaults to whichever `/models` would list first.

The corpus is the docs reachable from `AGENT_MAESTRO_ROOT` (no default —
setup.py writes it; see Config) — `CLAUDE.md`, `docs/*.md`, `core/README.md`,
each app's `README.md`, and this app's own `docs/*.md`: 98,717 bytes across
11 documents. It is read and embedded once at startup, not re-read per
request; `POST /index/rebuild` above is the way to pick up an edit without
restarting the process.
# trigger-test 1787763773
