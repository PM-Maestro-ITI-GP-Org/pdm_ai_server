"""
The tool half of A2b (docs/SCOPE.md §6.2, §6.3): read and navigate tools the
model may call, plus the schema that describes them to a backend.

Only the two tiers cleared for A2b are here. `search_docs` and `read_file`
never leave this process -- they are plain functions over the corpus already
loaded. `navigate_to` is different: the thing it would act on (MessageBus)
lives in the Qt process, not here, so this tool does not perform navigation
-- it validates the request against the five real tab ids and hands back an
instruction for the Qt client to publish as `agent.navigate` (§6.2). A wrong
tab name is caught here rather than reaching MessageBus as a free string.

"Act" tools (run a scenario, fetch, OTA) are explicitly not here -- that is
A4, unchanged, per docs/SCOPE.md §8.
"""

from pathlib import Path

import config
from index import Index

# The five ids `shell/main.cpp` registers apps under (`registerApp({"id": ...})`).
# A hardcoded tuple, not a live read of shell/main.cpp: the whole reason this
# is a tool and not a raw string field is that the model must not be able to
# name a tab that doesn't exist, and the shell doesn't expose this list itself.
TAB_IDS = ("motor_control", "data_collection", "mlops", "ota", "agent")

# Which named UI elements a page actually listens for on `agent.highlight`
# (a follow-up MessageBus publish `shell/qml/Main.qml` fires after a
# `navigate_to` with a `section`, once the tab switch and crossfade have
# settled -- see that file). An allowlist for the same reason TAB_IDS is one:
# the model must not be able to name an element that doesn't exist and get a
# `navigate_to` that silently highlights nothing. Grows one entry at a time,
# each added the moment the destination page is actually wired to flash it --
# an empty list here means that tab has none yet, not that none are planned.
HIGHLIGHT_TARGETS = {
    # Surveyed 2026-08-23 -- three of four delegated via Herdr to opencode
    # agents (Ox Alpha) reading each page's real QML plus its README, one
    # (mlops, small enough) done by hand after that session came back empty
    # twice in a row. Every entry below is wired: an id on the container, a
    # BusSubscription on agent.highlight, and a pulsing glow-ring animation,
    # in that tab's own QML file(s).
    "motor_control": (
        "emergency_stop",   # outside the ScrollView, always on screen -- cheapest, most important
        "fetch_panel",      # the original A2b example: RIG_ACCESS.md's fetch-and-clear flow
        "scenario_grid",    # run one of the scripted A-J drive profiles
        "custom_sessions",  # record/replay a hand sweep
        "series_builder",   # queue several scenarios/sweeps as one campaign
    ),
    "data_collection": (
        "live_plot",        # the rolling trace of the current recording
        "live_telemetry",   # the 13-channel numeric readout
        "event_log",        # command results, warnings, errors
        "column_picker",    # choosing which signals a loaded CSV graphs (graph view)
    ),
    "mlops": (
        "run_gate",         # the verdict pill + Run gate/Reload controls
        "gate_checks",      # per-check pass/fail list -- needs a gate report to exist
        "metrics",          # the CNN/RUL MetricCards behind the verdict -- same caveat
    ),
    "ota": (
        "guest_list",        # Guests sub-screen (the default one) -- start/kill/info/shell a guest
        "update_stepper",    # OTA Update sub-screen -- the upload/pull/apply flow
        "send_files_panel",  # OTA Update sub-screen -- pushing arbitrary files into a guest
        "shell_terminal",    # Shell sub-screen -- one-shot exec or an interactive session
    ),
    "agent": (),
}

# Mirrors CLAUDE.md's "the seven repositories" table. Static because the repo
# list itself changes rarely; what *is* live is whether each Maestro app has
# been ported to the contract, checked from disk below rather than repeated
# here by hand.
_REPOS = [
    {"name": "PdM-Maestro_gui", "role": "Shell + submodule pins", "app_id": None},
    {"name": "pdm_app_core", "role": "Shared palette, message bus, app registry, "
                                      "broker settings, safety stop", "app_id": None},
    {"name": "pdm_motor_control_gui", "role": "Rig control tab", "app_id": "motor_control"},
    {"name": "motor_recorder_gui", "role": "Data Collection tab", "app_id": "data_collection"},
    {"name": "pdm_mlops_gui", "role": "ML/Ops tab", "app_id": "mlops"},
    {"name": "ota_update_gui", "role": "OTA Update tab", "app_id": "ota"},
    {"name": "pdm_ai_agent_gui", "role": "AI Agent tab", "app_id": "agent"},
]


def list_repo() -> dict:
    """The repo graph, with each Maestro tab's live integration state.

    "Integrated" means `apps/<id>/pdm-app.cmake` exists -- the same marker
    Maestro's own CMake checks (docs/ARCHITECTURE.md, "Declaring compliance")
    -- so this answers from the actual checkout, not from a doc that can go
    stale the way docs/STATUS.md's ML/Ops account already has.
    """
    root = Path(config.AGENT_MAESTRO_ROOT)
    repos = []
    for repo in _REPOS:
        entry = dict(repo)
        if repo["app_id"] is not None:
            entry["integrated"] = (root / "apps" / repo["app_id"] / "pdm-app.cmake").exists()
        repos.append(entry)
    return {"repos": repos}


def read_file(path: str, corpus_paths: list[str]) -> dict:
    """Read one document from the agent's own corpus.

    Scoped to `corpus_paths` -- the same allowlist `/chat` retrieves over --
    rather than an arbitrary path under the repo root. The corpus is the part
    of the toolchain this tab is grounded in and cleared to explain; reading
    outside it (source files, `model_out/metrics.json`, recordings) is real
    and wanted (§6.2 calls file-backed state in scope) but is its own
    allowlist to design deliberately, not a side effect of this one.
    """
    if path not in corpus_paths:
        return {"error": f"{path!r} is not in the agent's corpus"}
    root = Path(config.AGENT_MAESTRO_ROOT)
    try:
        return {"path": path, "text": (root / path).read_text()}
    except OSError as exc:
        return {"error": f"could not read {path}: {exc}"}


async def search_docs(query: str, index: Index, top_k: int | None = None) -> dict:
    """The same retrieval `/chat` always runs, exposed as a tool the model can
    re-call with a different query -- the one already injected does not need
    a tool, this is for when it turns out not to be enough."""
    try:
        hits = await index.search(query, top_k)
    except Exception as exc:  # index not ready, embedding backend down, etc.
        return {"error": str(exc)}
    return {
        "results": [
            {"citation": chunk.citation, "text": chunk.text, "score": round(score, 4)}
            for chunk, score in hits
        ]
    }


def navigate_to(tab: str, section: str = "") -> dict:
    """Validate, don't perform. See the module docstring for why."""
    if tab not in TAB_IDS:
        return {"error": f"{tab!r} is not a tab; must be one of {TAB_IDS}"}
    if section and section not in HIGHLIGHT_TARGETS.get(tab, ()):
        known = HIGHLIGHT_TARGETS.get(tab, ())
        return {"error": f"{section!r} is not a highlightable element on {tab!r}; "
                          f"known elements there: {known or '(none yet)'}"}
    return {"navigate": {"tab": tab, "section": section}}


# OpenAI-style function-calling schema, the format both backends accept --
# llama-server via --jinja, Ollama natively (docs/SCOPE.md §6.3's "a model
# that is actually competent at structured tool use").
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the PdM Maestro documentation corpus for sections "
                            "relevant to a query. Use this to look something up beyond "
                            "what's already in the sources you were given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what to search for"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one full document from the corpus by its path, e.g. "
                            "'docs/STATUS.md'. Use when a retrieved excerpt isn't enough "
                            "context and you need the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "repo-relative path, "
                                                                "exactly as a citation shows it"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_repo",
            "description": "List the seven repositories in the PdM toolchain, what each "
                            "one is, and whether each Maestro tab is actually integrated "
                            "into the build right now.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Switch the running app to a given tab. This only moves the UI "
                            "-- it cannot change system or hardware state. Call it whenever "
                            "the answer explains how to use something that lives on a "
                            "specific tab, not only when asked outright to go there -- the "
                            "point is to leave the user looking at the thing just explained, "
                            "not just told about it. Optionally also names one UI element on "
                            "that tab to highlight (flash) once the switch lands, walking the "
                            "user's eye to the exact control the answer was about -- only use "
                            "one of the known elements listed for that tab; leave it out "
                            "entirely rather than guess at a name. Known elements per tab: "
                            + "; ".join(f"{tab}: {', '.join(elements) or '(none yet)'}"
                                        for tab, elements in HIGHLIGHT_TARGETS.items()),
            "parameters": {
                "type": "object",
                "properties": {
                    "tab": {"type": "string", "enum": list(TAB_IDS)},
                    "section": {"type": "string", "description": "the UI element to highlight "
                                                                   "on that tab, from the list "
                                                                   "above -- omit if none apply"},
                },
                "required": ["tab"],
            },
        },
    },
]


async def dispatch(name: str, arguments: dict, *, index: Index, corpus_paths: list[str]) -> dict:
    """Run one tool call by name. Never raises -- a bad call is a result the
    model (or the fallback) can see, not a crashed request."""
    try:
        if name == "search_docs":
            return await search_docs(arguments["query"], index, arguments.get("top_k"))
        if name == "read_file":
            return read_file(arguments["path"], corpus_paths)
        if name == "list_repo":
            return list_repo()
        if name == "navigate_to":
            return navigate_to(arguments["tab"], arguments.get("section", ""))
        return {"error": f"unknown tool {name!r}"}
    except (KeyError, TypeError) as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
