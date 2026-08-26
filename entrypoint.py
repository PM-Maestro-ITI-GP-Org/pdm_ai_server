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
from pathlib import Path

import config


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
