"""
Easy setup for the AI Agent server, runnable straight from a fresh checkout:

    python3 setup.py

A numbered-prompt walk through exactly the things this project otherwise makes
you do by hand (docs/SCOPE.md §6): point the server at a Maestro checkout
(cloning one by default if run standalone, e.g. from a release tarball),
pick a backend and a model (downloaded right here, so the AI Agent tab has
one ready without a trip through its Settings view first), create the venv,
and optionally build llama-server right here (the multi-minute
git-clone-and-compile that runtime.py automates over HTTP once the server is
up -- at setup time nobody is up yet, so the wizard calls into runtime.py
directly).

Everything it decides is written to one TOML file (~/.config/pdm_agent/
config.toml, or AGENT_CONFIG_PATH) that config.py already reads, env vars
still winning -- so a systemd unit or docker file can override any answer
given here without editing the file underneath.

It also writes the Qt side's two settings (server URL + the command behind the
"Start local AI" button in the AI Agent tab) into the PdM Maestro QSettings
ini, so after this script the GUI needs zero configuration. Refusing to touch
that ini silently when the Maestro conf directory doesn't exist would leave
the button dead with no explanation, so a missing file is created instead --
QSettings reads whatever is there on next launch.

Every prompt takes its default from Enter alone; Ctrl-C exits at any point.
"""

import asyncio
import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path

import config

SERVER_DIR = Path(__file__).resolve().parent

# Where the Qt shell keeps its settings. Must match pdm_ai_agent_gui's
# shell/main.cpp setOrganizationName/setApplicationName exactly -- these
# strings are the contract between the two repos, and nothing else in this
# one derives them programmatically.
QT_CONFIG_DIR = Path("~/.config/PM-Maestro-ITI-GP-Org").expanduser()
QT_CONFIG_FILE = QT_CONFIG_DIR / "PdM Maestro.conf"

# A release tarball (this file, downloaded standalone) has no Maestro
# checkout to walk up to. Clone one here by default instead of making every
# fresh install hand-type a path.
MAESTRO_REPO_URL = "https://github.com/PM-Maestro-ITI-GP-Org/PdM-Maestro_gui.git"
DEFAULT_MAESTRO_ROOT = Path("~/.local/share/pdm_agent/PdM-Maestro_gui").expanduser()


def ask(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def choose(prompt: str, options: list[str], default_index: int = 0) -> int:
    numbered = ", ".join(f"{i + 1}) {o}" for i, o in enumerate(options))
    raw = input(f"{prompt} ({numbered}) [{options[default_index]}]: ").strip()
    if not raw:
        return default_index
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return int(raw) - 1
    print(f"  -> '{raw}' is not one of the choices; using {options[default_index]}")
    return default_index


def detect_maestro_root() -> str:
    """The checkout this wizard usually runs inside, found the same way
    test_tools.py finds it: walk up looking for CLAUDE.md."""
    for parent in SERVER_DIR.parents:
        if (parent / "CLAUDE.md").exists():
            return str(parent)
    if (DEFAULT_MAESTRO_ROOT / "CLAUDE.md").exists():
        return str(DEFAULT_MAESTRO_ROOT)
    return ""


def clone_maestro(dest: str) -> bool:
    """Fetch a fresh checkout for a standalone install that isn't running
    from inside one already. Needs submodules -- the corpus reads app repo
    docs (e.g. apps/motor_control/README.md) straight out of them."""
    if not shutil.which("git"):
        print("  ! git not on PATH -- can't clone; check out "
              f"{MAESTRO_REPO_URL} by hand and rerun setup with that path")
        return False
    print(f"  cloning {MAESTRO_REPO_URL}\n    -> {dest} (with submodules, this takes a bit) ...")
    try:
        subprocess.run(
            ["git", "clone", "--recurse-submodules", MAESTRO_REPO_URL, dest],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"  ! clone failed ({exc}); check out {MAESTRO_REPO_URL} by hand "
              "and rerun setup with that path")
        return False
    return True


# ---- model picker ----------------------------------------------------------

def choose_model() -> dict | None:
    """Pick one of config.CATALOG's four curated models, or skip. Same list
    the AI Agent tab's Settings view offers -- doing it here means a fresh
    install can be ready to chat without opening the GUI first."""
    labels = [m["label"] for m in config.CATALOG]
    idx = choose("Which model should this box use", labels + ["skip for now"])
    if idx == len(config.CATALOG):
        return None
    return config.CATALOG[idx]


def download_model(entry: dict) -> None:
    """Synchronous fetch with a terminal progress bar, same URL and
    destination convention as main.py's own /download -- duplicated instead
    of imported because main.py can't be imported before AGENT_MAESTRO_ROOT
    is written (it builds the corpus at module scope)."""
    dest_dir = Path(config.AGENT_MODELS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / entry["filename"]
    if final_path.exists() and final_path.stat().st_size == entry["size_bytes"]:
        print(f"  already downloaded: {final_path}")
        return

    import httpx  # only needed once an actual fetch is happening

    partial_path = dest_dir / f"{entry['filename']}.partial"
    url = f"https://huggingface.co/{entry['repo']}/resolve/main/{entry['filename']}"
    print(f"  downloading {entry['label']} -> {final_path}")
    downloaded = 0
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=None) as resp:
            resp.raise_for_status()
            with open(partial_path, "wb") as f:
                for chunk in resp.iter_bytes(1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"\r    {downloaded / entry['size_bytes'] * 100:5.1f}%", end="", flush=True)
        print()
        if downloaded != entry["size_bytes"]:
            raise ValueError(f"size mismatch: got {downloaded}, expected {entry['size_bytes']}")
        partial_path.rename(final_path)
        print(f"  done: {final_path}")
    except Exception as exc:  # network error, disk full, size mismatch
        print(f"\n  ! download failed ({exc}) -- retry later from the AI Agent "
              "tab's Settings view")
        partial_path.unlink(missing_ok=True)


# ---- starting the backend --------------------------------------------------

async def _wait_reachable(port: int, timeout: float) -> bool:
    """Poll llama-server's own /v1/models the same way main.py's /health
    does, since starting the process is not the same as it being loaded and
    answering -- a 7B model can take real time to come up."""
    import httpx

    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient(timeout=2.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(f"http://localhost:{port}/v1/models")
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            print(".", end="", flush=True)
            await asyncio.sleep(2.0)
    return False


def start_backend_and_wait(model_entry: dict, timeout: float = 120.0) -> bool:
    """Start llama-server on the downloaded model and the standard embed
    model, then wait for both to actually answer -- so setup ends with the
    server genuinely connected, not just process-started. Requires the
    runtime binary (offer_runtime_build) to already exist; returns False
    without doing anything otherwise, same as /runtime/backend/start would.

    ponytail: this re-downloads the same .gguf a second time, into
    llama-server's own -hf cache -- start_chat_backend takes a repo:quant id,
    not the file setup.py already fetched into AGENT_MODELS_DIR. One-time
    cost (llama-server caches it after), not a fix for this pass; pointing
    it at the local file directly (`-m <path>`) is the upgrade if the
    duplicate download ever matters enough to justify touching runtime.py's
    contract.
    """
    import runtime

    if not runtime.is_available():
        print("  ! no llama-server binary -- skipped; POST /runtime/build "
              "once the server is running, or rerun this wizard")
        return False

    hf_id = f"{model_entry['repo']}:{model_entry['quant']}"

    async def _run() -> bool:
        await runtime.start_chat_backend(hf_id, port=8080)
        await runtime.start_embed_backend(config.EMBED_MODEL_HF_ID, port=8081)
        print(f"  waiting for {model_entry['label']} to load ", end="", flush=True)
        ok = await _wait_reachable(8080, timeout)
        print(" ready" if ok else " timed out")
        return ok

    return asyncio.run(_run())


# ---- 1. the config file --------------------------------------------------

def write_config(maestro_root: str, backend: str, port: int, active_model: str = "") -> Path:
    """Merge the four answers into the existing file, keeping every key this
    script doesn't manage -- rerunning setup after hand-editing something
    else must not erase that edit."""
    existing: dict = {}
    if config.CONFIG_PATH.exists():
        import tomllib
        try:
            with open(config.CONFIG_PATH, "rb") as f:
                existing = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            print(f"  ! {config.CONFIG_PATH} is unreadable and will be replaced")

    managed = {
        "AGENT_MAESTRO_ROOT": maestro_root,
        "AGENT_BACKEND": backend,
        "AGENT_SERVER_PORT": port,
        "AGENT_ACTIVE_MODEL": active_model,
    }
    merged = {**existing, **managed}

    def fmt(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'

    config.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Written by setup.py -- safe to edit by hand; environment variables",
        "# still override anything in here (see config.py for every key).",
    ]
    lines += [f"{key} = {fmt(merged[key])}" for key in sorted(merged)]
    config.CONFIG_PATH.write_text("\n".join(lines) + "\n")
    return config.CONFIG_PATH


# ---- 2. python environment ----------------------------------------------

def ensure_venv() -> Path | None:
    venv_python = SERVER_DIR / "venv" / "bin" / "python"
    if venv_python.exists():
        print("  venv already present")
    elif shutil.which("python3") and input(
            "  No venv found. Create one and install requirements.txt now? [Y/n]: ").strip().lower() in ("", "y"):
        subprocess.run([sys.executable, "-m", "venv", str(SERVER_DIR / "venv")], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install",
                        "-r", str(SERVER_DIR / "requirements.txt")], check=True)
        print(f"  created {SERVER_DIR / 'venv'}")
    else:
        print("  ! skipping -- the server needs `fastapi uvicorn httpx` "
              "(requirements.txt) however you choose to install them")
        return None

    missing = []
    for module in ("fastapi", "uvicorn", "httpx"):
        probe = subprocess.run([str(venv_python), "-c", f"import {module}"],
                               capture_output=True)
        if probe.returncode != 0:
            missing.append(module)
    if missing:
        print(f"  ! venv is missing {', '.join(missing)} -- run "
              f"`{venv_python.parent / 'pip'} install -r requirements.txt`")
    return venv_python


# ---- 3. llama-server (optional, multi-minute) ----------------------------

def offer_runtime_build() -> None:
    import runtime

    if runtime.is_available():
        print(f"  llama-server already built: {runtime.binary_path()}")
        return
    if not shutil.which("git") or not shutil.which("cmake"):
        print("  ! skipping build -- git and cmake must be on PATH; the server "
              "can do it later via POST /runtime/build once running")
        return
    answer = input(
        "  Build llama-server from source now? Downloads and compiles\n"
        f"  into {config.AGENT_RUNTIME_DIR}; several minutes on a laptop,\n"
        "  CUDA enabled automatically where nvcc exists. [y/N]: ").strip().lower()

    async def run_with_progress() -> None:
        task = asyncio.create_task(runtime._run_build())
        while not task.done():
            await asyncio.sleep(1.0)
            stage = runtime._build_state["stage"]
            tail = runtime._build_state["log_tail"][-1:] if runtime._build_state["log_tail"] else []
            print(f"\r  [{stage}] {' '.join(tail)[:100]:<100}", end="", flush=True)
        print()
        if runtime._build_state["error"]:
            print(f"  ! build failed: {runtime._build_state['error']}")
            print("    the full log tail is in GET /runtime/build/status once the server runs")
        else:
            print(f"  built: {runtime.binary_path()}")

    if answer == "y":
        asyncio.run(run_with_progress())
    else:
        print("  skipped -- the server (or the wizard, rerun anytime) can build later")


# ---- 4. the Qt side -------------------------------------------------------

def write_qt_settings(url: str, start_command: str) -> None:
    parser = configparser.RawConfigParser(strict=False)
    parser.optionxform = str  # QSettings keys are case-sensitive ("serverUrl")
    if QT_CONFIG_FILE.exists():
        parser.read(QT_CONFIG_FILE, encoding="utf-8")
    if not parser.has_section("agent"):
        parser.add_section("agent")
    parser.set("agent", "serverUrl", url)
    parser.set("agent", "serverStartCommand", start_command)
    QT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(QT_CONFIG_FILE, "w", encoding="utf-8") as f:
        parser.write(f, space_around_delimiters=False)
    print(f"  wrote {QT_CONFIG_FILE} ([agent] serverUrl / serverStartCommand)")


def main() -> int:
    print("== PdM AI Agent server -- setup ==\n")

    detected = detect_maestro_root()
    maestro_root = ask(
        "Where is the PdM Maestro checkout (the folder containing CLAUDE.md)?",
        detected or str(DEFAULT_MAESTRO_ROOT),
    )
    maestro_root = str(Path(maestro_root).expanduser().resolve())
    if not (Path(maestro_root) / "CLAUDE.md").exists():
        if not Path(maestro_root).exists():
            clone_maestro(maestro_root)
        if not (Path(maestro_root) / "CLAUDE.md").exists():
            print(f"  ! {maestro_root} does not look like a Maestro checkout "
                  f"(no CLAUDE.md) -- written anyway; the server will refuse to start on it")

    backend = ["llamacpp", "ollama"][choose("Which model backend?", ["llamacpp", "ollama"])]
    port = int(ask("Port for this server", str(config._get("AGENT_SERVER_PORT", "8420"))))

    # Ollama manages its own model store (`ollama pull`) -- CATALOG is only
    # the raw .gguf files llama-server loads directly.
    model_entry = choose_model() if backend == "llamacpp" else None

    path = write_config(maestro_root, backend, port, model_entry["id"] if model_entry else "")
    print(f"\n[1/5] config written: {path}")

    print("[2/5] python environment:")
    venv_python = ensure_venv()

    print("[3/5] llama-server runtime:")
    offer_runtime_build()

    print("[4/6] model:")
    if model_entry:
        download_model(model_entry)
    else:
        print("  skipped -- pick one later from the AI Agent tab's Settings view")

    print("[5/6] starting the backend:")
    backend_ready = model_entry is not None and start_backend_and_wait(model_entry)
    if model_entry and not backend_ready:
        print("  ! not connected -- start it later from the AI Agent tab, or "
              "POST /runtime/backend/start once the server is running")

    print("[6/6] Qt side:")
    url = f"http://127.0.0.1:{port}"
    if venv_python:
        # exec so the launched server replaces the shell, not outlives it as
        # an extra process; cwd matters because uvicorn resolves main.py
        # relative to it.
        command = (f"cd '{SERVER_DIR}' && exec '{venv_python.parent / 'uvicorn'}' "
                   f"main:app --host 127.0.0.1 --port {port}")
        write_qt_settings(url, command)
    else:
        print("  ! skipped -- no venv, so there is no uvicorn to point the "
              "'Start local AI' button at; set agent/serverStartCommand by hand in")
        print(f"    {QT_CONFIG_FILE}")

    print(
        "\nDone. Next steps:\n"
        f"  * launch the PdM Maestro app -> AI Agent tab -> 'Start local AI'\n"
        f"    (or by hand: cd {SERVER_DIR} && ./venv/bin/uvicorn main:app --host 127.0.0.1 --port {port})\n"
        f"  * server URL: {url}\n"
        + (f"  * model ready and serving: {model_entry['label']}\n" if backend_ready else
           f"  * model ready: {model_entry['label']} -- backend didn't come up, see above\n" if model_entry else
           "  * no model downloaded -- pick one from inside the app (Settings tab of the AI Agent)\n")
        + "    if step 3 was skipped, POST /runtime/build builds llama-server on demand"
    )
    return 0


def run_frozen_setup() -> None:
    """Same wizard, trimmed for entrypoint.py's frozen-executable path: no
    venv step (dependencies are already bundled into the binary), and the
    Qt "Start local AI" command points straight at this executable
    (sys.executable, which PyInstaller's bootloader resolves to the running
    binary itself) instead of a venv's uvicorn."""
    print("== PdM AI Agent server -- first-run setup ==\n")

    detected = detect_maestro_root()
    maestro_root = ask(
        "Where is the PdM Maestro checkout (the folder containing CLAUDE.md)?",
        detected or str(DEFAULT_MAESTRO_ROOT),
    )
    maestro_root = str(Path(maestro_root).expanduser().resolve())
    if not (Path(maestro_root) / "CLAUDE.md").exists():
        if not Path(maestro_root).exists():
            clone_maestro(maestro_root)
        if not (Path(maestro_root) / "CLAUDE.md").exists():
            print(f"  ! {maestro_root} does not look like a Maestro checkout "
                  f"(no CLAUDE.md) -- written anyway; the server will refuse to start on it")

    backend = ["llamacpp", "ollama"][choose("Which model backend?", ["llamacpp", "ollama"])]
    port = int(ask("Port for this server", str(config._get("AGENT_SERVER_PORT", "8420"))))
    model_entry = choose_model() if backend == "llamacpp" else None

    path = write_config(maestro_root, backend, port, model_entry["id"] if model_entry else "")
    print(f"\n[1/4] config written: {path}")

    print("[2/4] llama-server runtime:")
    offer_runtime_build()

    print("[3/5] model:")
    if model_entry:
        download_model(model_entry)
    else:
        print("  skipped -- pick one later from the AI Agent tab's Settings view")

    print("[4/5] starting the backend:")
    backend_ready = model_entry is not None and start_backend_and_wait(model_entry)
    if model_entry and not backend_ready:
        print("  ! not connected -- POST /runtime/backend/start once the "
              "server is running, or check GET /runtime/build/status")

    print("[5/5] Qt side:")
    url = f"http://127.0.0.1:{port}"
    write_qt_settings(url, f"exec '{sys.executable}'")

    print(
        "\nSetup done -- starting the server now.\n"
        f"  * server URL: {url}\n"
        + (f"  * model ready and serving: {model_entry['label']}\n" if backend_ready else
           f"  * model downloaded: {model_entry['label']} -- backend didn't come up, see above\n" if model_entry else
           "  * no model downloaded -- pick one from the AI Agent tab's Settings view\n")
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\naborted -- nothing half-written: each step above is atomic")
        raise SystemExit(130)
