"""
Self-check for runtime.py: the build-guard and process-tracking logic,
without an actual multi-minute git-clone-and-compile or a real llama-server
process. Plain asserts, no framework: run with `python3 test_runtime.py`.
"""

import asyncio

import runtime


def test_is_available_false_when_nothing_present(monkeypatch):
    monkeypatch.setattr(runtime, "binary_path", lambda: __import__("pathlib").Path("/nonexistent"))
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    assert runtime.is_available() is False


def test_resolve_binary_prefers_runtime_dir_over_path(monkeypatch):
    fake_own = type("P", (), {"exists": lambda self: True, "__str__": lambda self: "/runtime/dir/llama-server"})()
    monkeypatch.setattr(runtime, "binary_path", lambda: fake_own)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/bin/llama-server")
    assert runtime.resolve_binary() == "/runtime/dir/llama-server"


def test_resolve_binary_raises_when_nothing_available(monkeypatch):
    fake_missing = type("P", (), {"exists": lambda self: False})()
    monkeypatch.setattr(runtime, "binary_path", lambda: fake_missing)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    try:
        runtime.resolve_binary()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "runtime/build" in str(exc)


def test_start_build_refuses_concurrent_builds(monkeypatch):
    # Swap in a coroutine that never actually clones/compiles anything --
    # this test is about the single-build guard, not the build itself.
    async def fake_build():
        await asyncio.sleep(10)
    monkeypatch.setattr(runtime, "_run_build", fake_build)
    runtime._build_state.update(active=False, stage="", log_tail=[], done=False, error=None)

    async def run():
        first = runtime.start_build()
        second = runtime.start_build()
        return first, second

    first, second = asyncio.run(run())
    assert first is True
    assert second is False
    runtime._build_state["active"] = False  # don't leak state into other tests


def test_backend_running_reflects_process_state():
    runtime._processes.clear()

    class FakeProc:
        def __init__(self, returncode):
            self.returncode = returncode

    assert runtime.backend_running("chat") is False  # nothing tracked yet

    runtime._processes["chat"] = FakeProc(returncode=None)  # still running
    assert runtime.backend_running("chat") is True

    runtime._processes["chat"] = FakeProc(returncode=0)  # exited
    assert runtime.backend_running("chat") is False

    runtime._processes.clear()


class _MonkeyPatch:
    """Same hand-rolled shim test_tools.py uses -- not worth a pytest
    dependency for one fixture shared by two test files."""

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
