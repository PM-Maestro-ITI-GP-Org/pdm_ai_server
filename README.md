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
