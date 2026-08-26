"""
Entry point for the standalone executable (built by CI's PyInstaller step).

main.py refuses to import at all without a valid AGENT_MAESTRO_ROOT -- correct
for `uvicorn main:app`, where setup.py has always already run. A person who
just downloaded the frozen executable has run neither, so this checks
readiness first and runs setup.py's wizard (clone the Maestro checkout, pick
a backend, download a model) before main.py's module-level corpus-building
ever executes. `python main.py` and `uvicorn main:app` are untouched.
"""
import importlib
import shutil
import sys
from pathlib import Path

import config


def _cpu_fallback_marker() -> Path:
    """Sits next to runtime.binary_path() only while that file is the
    bundled CPU-only fallback, not a real machine-specific build -- lets
    _maybe_upgrade_to_cuda tell the two apart without guessing from the
    binary's contents."""
    import runtime

    return runtime.binary_path().parent / ".bundled_cpu_fallback"


def _extract_bundled_runtime() -> None:
    """CI's release job bundles a CPU-only llama-server inside this
    executable (see ci.yml's release job) so a fresh download has a working
    binary with zero local build step. Copies it out to AGENT_RUNTIME_DIR
    once; every launch after finds it already there and skips this. A
    source install (`python entrypoint.py`, no PyInstaller) has nothing to
    extract from -- sys._MEIPASS only exists inside a frozen process."""
    if not getattr(sys, "frozen", False) or not hasattr(sys, "_MEIPASS"):
        return
    import runtime

    if runtime.binary_path().exists():
        return
    bundled = Path(sys._MEIPASS) / "llama-server-bin" / "llama-server"
    if not bundled.exists():
        return
    dest = runtime.binary_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundled, dest)
    dest.chmod(0o755)
    _cpu_fallback_marker().touch()
    print(f"using bundled llama-server: {dest}")


def _maybe_upgrade_to_cuda() -> None:
    """If the binary in place is still the bundled CPU fallback and this
    machine actually has CUDA, replace it with a real build automatically
    -- no prompt, since a CPU-only server on a CUDA box is leaving real
    speed on the table for no reason. One-time cost (several minutes) the
    first launch this is true; every launch after is a no-op fast marker
    check, since the marker is gone once the upgrade succeeds."""
    marker = _cpu_fallback_marker()
    if not marker.exists():
        return
    import setup
    if setup.build_for_cuda_if_available():
        marker.unlink(missing_ok=True)


def _config_ready() -> bool:
    return bool(config.AGENT_MAESTRO_ROOT) and (Path(config.AGENT_MAESTRO_ROOT) / "CLAUDE.md").exists()


def _active_model() -> dict | None:
    """The model to bring up this launch, or None if there isn't one to try.

    Prefers AGENT_ACTIVE_MODEL (what the wizard just picked), but falls back
    to whatever catalog model is actually sitting in AGENT_MODELS_DIR --
    covers a config.toml written before this key existed, or one where the
    recorded id was since deleted from disk -- and persists the find so
    later launches don't need to repeat this scan.
    """
    if config.BACKEND != "llamacpp":
        return None  # ollama manages its own models; nothing for us to start

    models_dir = Path(config.AGENT_MODELS_DIR)

    if config.AGENT_ACTIVE_MODEL:
        entry = next((m for m in config.CATALOG if m["id"] == config.AGENT_ACTIVE_MODEL), None)
        if entry and (models_dir / entry["filename"]).exists():
            return entry

    for entry in config.CATALOG:
        path = models_dir / entry["filename"]
        if path.exists() and path.stat().st_size == entry["size_bytes"]:
            import setup
            setup.write_config(config.AGENT_MAESTRO_ROOT, config.BACKEND,
                                config.AGENT_SERVER_PORT, entry["id"])
            return entry
    return None


def main() -> int:
    _extract_bundled_runtime()
    _maybe_upgrade_to_cuda()

    # Gated on config alone, not on a model being downloaded -- a deliberate
    # "skip for now" during the wizard must not turn into the wizard
    # re-running on every subsequent launch. Matches the source install's
    # behavior: skipping the model there doesn't force setup.py to rerun
    # either, the AI Agent tab's Settings view is the path to get one later.
    ran_wizard = False
    if not _config_ready():
        import setup
        setup.run_frozen_setup()
        importlib.reload(config)  # pick up what the wizard just wrote to disk
        ran_wizard = True

    # The wizard already starts the backend itself, with progress output,
    # when it just ran. Every launch after that is a brand new process with
    # no llama-server child of its own yet -- config being valid doesn't
    # mean anything is actually running, so this tries again here, quietly.
    if not ran_wizard:
        entry = _active_model()
        if entry is not None:
            import setup
            print(f"starting {entry['label']}...")
            setup.start_backend_and_wait(entry)

    from main import app  # deferred: main.py builds the corpus at import time
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.AGENT_SERVER_PORT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
