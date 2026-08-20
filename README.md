# pdm_ai_agent server

The Python side of the AI Agent tab (`docs/SCOPE.md` §6.1). Runs outside Qt,
talks to a local model backend, and is reached over HTTP by the Qt client.
Phase A2a: health and model listing only, no chat yet.

## Install

```bash
cd apps/agent/server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port ${AGENT_SERVER_PORT:-8420}
```

## Config

Environment variables, all optional:

| Variable | Default | |
|---|---|---|
| `AGENT_BACKEND` | `llamacpp` | `llamacpp` or `ollama` |
| `LLAMACPP_HOST` | `http://localhost:8080` | used when `AGENT_BACKEND=llamacpp` |
| `OLLAMA_HOST` | `http://localhost:11434` | used when `AGENT_BACKEND=ollama` |
| `AGENT_SERVER_PORT` | `8420` | |
| `AGENT_MODELS_DIR` | `~/.cache/pdm_agent/models` | downloaded `.gguf` files land here, never in the repo |

Point the relevant host at a second laptop to use it as the model server —
no code change (`docs/SCOPE.md` §3.3, §6.4).

**Why two backends:** `llama-server` loads exactly one model per process —
there's no multi-model registry to list, so under `llamacpp`, "which model"
means "which server," and switching models means pointing `LLAMACPP_HOST` at
a different running `llama-server`. Ollama can list and switch between many
pulled models from one process. Both are supported because the models are
meant to be genuinely interchangeable across machines, not tied to whichever
runtime happens to be installed on this one (`docs/SCOPE.md` §6.4).

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
