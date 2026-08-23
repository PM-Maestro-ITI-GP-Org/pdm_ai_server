"""
Self-check for the A2b tool half: tools.py's own functions, the two tool-call
wire formats main.py has to read (llama-server's JSON-string arguments,
Ollama's already-parsed dict), and one full /chat round trip through a
stand-in model backend and a stand-in index -- no real backend, no real
embeddings, no network. Same shape server/README.md already claims for
retrieval ("exercised in tests with a stand-in embedder").

Plain asserts, no framework: run with `python3 test_tools.py`.
"""

import asyncio
import os
from pathlib import Path


def _require_maestro_root() -> None:
    """main.py refuses to start without a real Maestro checkout (config.py's
    AGENT_MAESTRO_ROOT has no default) -- that guard is correct for the server
    and hostile for tests. Before importing it, honor the environment var or
    detect the checkout this file usually sits inside (apps/agent/server is
    three levels under the Maestro root), so `python3 test_tools.py` works in
    place; only a standalone clone of server/ with no tree above it has to set
    anything by hand."""
    if os.environ.get("AGENT_MAESTRO_ROOT"):
        return
    for parent in Path(__file__).resolve().parents:
        if (parent / "CLAUDE.md").exists():
            os.environ["AGENT_MAESTRO_ROOT"] = str(parent)
            return
    raise SystemExit(
        "test_tools.py: AGENT_MAESTRO_ROOT is not set and no PdM Maestro "
        "checkout was found above this file -- export it and rerun."
    )


_require_maestro_root()

import config  # noqa: E402  (needs the env var above set first)
import main  # noqa: E402
import tools  # noqa: E402
from corpus import Chunk  # noqa: E402


def test_navigate_to_valid():
    result = tools.navigate_to("agent")
    assert result == {"navigate": {"tab": "agent", "section": ""}}, result


def test_navigate_to_with_known_highlight_element():
    result = tools.navigate_to("motor_control", "fetch_panel")
    assert result == {"navigate": {"tab": "motor_control", "section": "fetch_panel"}}, result


def test_navigate_to_rejects_unknown_highlight_element():
    result = tools.navigate_to("motor_control", "warp_drive")
    assert "error" in result, result
    # A tab with zero wired elements gets a clearer message than a bare list.
    result_empty_tab = tools.navigate_to("agent", "chat")
    assert "error" in result_empty_tab, result_empty_tab


def test_navigate_to_rejects_unknown_tab():
    result = tools.navigate_to("dashboard")
    assert "error" in result, result


def test_every_declared_highlight_target_validates():
    # Cheap insurance against a typo in HIGHLIGHT_TARGETS itself: every name
    # it declares must actually pass navigate_to's own check for its tab, or
    # the model would be offered an element navigate_to then rejects.
    for tab, elements in tools.HIGHLIGHT_TARGETS.items():
        for element in elements:
            result = tools.navigate_to(tab, element)
            assert "navigate" in result, (tab, element, result)


def test_read_file_respects_allowlist():
    allowed = ["docs/STATUS.md"]
    ok = tools.read_file("docs/STATUS.md", allowed)
    assert "text" in ok and ok["path"] == "docs/STATUS.md", ok

    # Not merely "outside the repo" -- outside the *allowlist*, which is what
    # actually stops path traversal here: the check is membership in
    # corpus_paths, not string sanitization, so this rejects on its own.
    blocked = tools.read_file("../../../../etc/passwd", allowed)
    assert "error" in blocked, blocked


def test_list_repo_reports_live_integration():
    result = tools.list_repo()
    repos = {r["name"]: r for r in result["repos"]}
    assert len(repos) == 7, repos
    # All five Maestro apps carry pdm-app.cmake as of this checkout (agent's
    # landed with A1) -- see CLAUDE.md's own table.
    for name in ("pdm_motor_control_gui", "motor_recorder_gui", "pdm_mlops_gui",
                 "ota_update_gui", "pdm_ai_agent_gui"):
        assert repos[name]["integrated"] is True, repos[name]
    assert "integrated" not in repos["PdM-Maestro_gui"]


def test_tool_call_args_openai_style_json_string():
    raw = {"function": {"name": "navigate_to", "arguments": '{"tab": "agent"}'}}
    name, args = main._tool_call_name_and_args(raw)
    assert (name, args) == ("navigate_to", {"tab": "agent"}), (name, args)


def test_tool_call_args_ollama_style_dict():
    raw = {"function": {"name": "list_repo", "arguments": {}}}
    name, args = main._tool_call_name_and_args(raw)
    assert (name, args) == ("list_repo", {}), (name, args)


def test_tool_call_args_malformed_json_is_none_not_a_crash():
    raw = {"function": {"name": "navigate_to", "arguments": "{not json"}}
    name, args = main._tool_call_name_and_args(raw)
    assert name == "navigate_to" and args is None, (name, args)


class _FakeIndex:
    """Stands in for index.Index: fixed hits, fixed vectors, no embedding call."""

    def __init__(self, hits):
        self.ready = True
        self._hits = hits

    async def search(self, _question, _top_k=None):
        return self._hits

    def vector_of(self, _chunk):
        return [1.0, 0.0]


def _install_fake_index(hits):
    main.INDEX = _FakeIndex(hits)
    main.CHUNKS = [c for c, _ in hits]


async def _fake_embed_documents(texts):
    return [[1.0, 0.0] for _ in texts]


def _reset_tools_state():
    main._tools_state = {"disabled": False, "parse_failures": 0}


def test_chat_runs_one_tool_round_then_answers(monkeypatch):
    _reset_tools_state()
    chunk = Chunk(path="apps/motor_control/README.md", heading="State",
                  text="Motor control talks to the ESP32 rig.")
    _install_fake_index([(chunk, 0.9)])
    monkeypatch.setattr(main.embeddings, "embed_documents", _fake_embed_documents)

    calls = []

    async def fake_chat_message(messages, model, tool_schema):
        calls.append((len(messages), tool_schema is not None))
        if tool_schema is not None:
            # First turn: the model asks to navigate before answering.
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {
                        "name": "navigate_to", "arguments": '{"tab": "motor_control"}'
                    }},
                ],
            }
        # Second turn: given the tool result, it answers and cites [1].
        return {"role": "assistant",
                "content": "Motor Control runs the scripted A-J profiles. [1]"}

    monkeypatch.setattr(main, "_chat_message", fake_chat_message)

    body = main.ChatRequest(message="what does motor control do?")
    response = asyncio.run(main.chat(body))

    assert calls == [(2, True), (4, False)], calls  # system+user, then +assistant+tool
    assert response["tool_calls"] == [
        {"name": "navigate_to", "arguments": {"tab": "motor_control"},
         "result": {"navigate": {"tab": "motor_control", "section": ""}}}
    ], response["tool_calls"]
    assert response["grounded"] is True
    assert response["sources"][0]["cited"] is True


def test_repeated_parse_failures_disable_tools_for_the_session(monkeypatch):
    _reset_tools_state()
    chunk = Chunk(path="docs/STATUS.md", heading="", text="placeholder")
    _install_fake_index([(chunk, 0.5)])
    monkeypatch.setattr(main.embeddings, "embed_documents", _fake_embed_documents)

    async def always_malformed(messages, model, tool_schema):
        if tool_schema is None:
            return {"role": "assistant", "content": "fallback answer, no citation needed here"}
        return {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "x", "function": {"name": "read_file", "arguments": "{bad"}}],
        }

    monkeypatch.setattr(main, "_chat_message", always_malformed)

    for _ in range(main.TOOL_PARSE_FAILURE_LIMIT):
        asyncio.run(
            main.chat(main.ChatRequest(message="anything"))
        )
    assert main._tools_state["disabled"] is True, main._tools_state

    # Once disabled, no schema should be offered on the next call at all.
    seen = {}

    async def record_schema(messages, model, tool_schema):
        seen["offered"] = tool_schema is not None
        return {"role": "assistant", "content": "answered without tools"}

    monkeypatch.setattr(main, "_chat_message", record_schema)
    asyncio.run(main.chat(main.ChatRequest(message="anything")))
    assert seen["offered"] is False, seen


def test_backend_rejecting_tools_disables_and_still_answers(monkeypatch):
    _reset_tools_state()
    chunk = Chunk(path="docs/STATUS.md", heading="", text="placeholder")
    _install_fake_index([(chunk, 0.5)])
    monkeypatch.setattr(main.embeddings, "embed_documents", _fake_embed_documents)

    async def rejects_then_answers(messages, model, tool_schema):
        if tool_schema is not None:
            raise main.ToolsUnsupportedError("400: tools field not recognized")
        return {"role": "assistant", "content": "plain answer, no tool support here"}

    monkeypatch.setattr(main, "_chat_message", rejects_then_answers)

    response = asyncio.run(
        main.chat(main.ChatRequest(message="anything"))
    )
    assert response["answer"] == "plain answer, no tool support here"
    assert main._tools_state["disabled"] is True


class _MonkeyPatch:
    """The one piece of pytest's API these tests actually use. Written by
    hand rather than adding pytest as a dependency for one fixture."""

    def __init__(self):
        self._restores = []

    def setattr(self, obj, name, value):
        self._restores.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._restores):
            setattr(obj, name, old)


def _run(test_fn):
    needs_patch = "monkeypatch" in test_fn.__code__.co_varnames[: test_fn.__code__.co_argcount]
    if not needs_patch:
        test_fn()
        return
    mp = _MonkeyPatch()
    try:
        test_fn(mp)
    finally:
        mp.undo()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            _run(t)
            print(f"ok   {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
