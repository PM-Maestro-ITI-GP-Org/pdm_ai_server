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


def main() -> int:
    # Gated on config alone, not on a model being downloaded -- a deliberate
    # "skip for now" during the wizard must not turn into the wizard
    # re-running on every subsequent launch. Matches the source install's
    # behavior: skipping the model there doesn't force setup.py to rerun
    # either, the AI Agent tab's Settings view is the path to get one later.
    if not _config_ready():
        import setup
        setup.run_frozen_setup()
        importlib.reload(config)  # pick up what the wizard just wrote to disk

    from main import app  # deferred: main.py builds the corpus at import time
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.AGENT_SERVER_PORT)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
