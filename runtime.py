"""
Bootstrapping the local inference runtime: building `llama-server` and
starting/stopping it as a child process, so a fresh checkout needs nothing
pre-installed beyond git, cmake, and a C/C++ toolchain -- the same tools a
Qt/C++ developer on this project already has.

Not a downloaded prebuilt binary. `ggml-org/llama.cpp`'s "latest" GitHub
release currently carries no usable asset (checked 2026-08-23: one 7-byte
`nightly-tag.txt`, nothing else) -- the project has moved to a nightly
release scheme this server would have to track and re-verify indefinitely.
Building from source instead is what this session already did by hand for
this exact GPU (docs/SCOPE.md's A2b real-hardware section) and is the only
approach that reliably targets whichever compute capability is actually
present, GTX 1650 through RTX 4070+ (docs/SCOPE.md §6.4), via
`-DCMAKE_CUDA_ARCHITECTURES=native` rather than a guess baked into a
prebuilt asset name.
"""

import asyncio
import shutil
from pathlib import Path

import config

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"

# One build at a time, same reasoning as _download_state in main.py: plain
# module state is safe because every mutation happens on the event loop with
# no `await` between check and set.
_build_state = {
    "active": False,
    "stage": "",       # "cloning" | "configuring" | "building" | "installing"
    "log_tail": [],     # last few lines, for a status endpoint to show progress
    "done": False,
    "error": None,
}

_LOG_TAIL_MAX = 20


def binary_path() -> Path:
    return Path(config.AGENT_RUNTIME_DIR) / "bin" / "llama-server"


def is_available() -> bool:
    """The runtime dir's own copy, or one already on PATH -- a developer who
    already has llama.cpp built and on PATH shouldn't be made to rebuild it."""
    if binary_path().exists():
        return True
    return shutil.which("llama-server") is not None


def resolve_binary() -> str:
    """What to actually exec. Prefers the runtime dir's own build over PATH,
    so a version this server built and knows the provenance of is used even
    if an older `llama-server` happens to be on PATH too."""
    own = binary_path()
    if own.exists():
        return str(own)
    found = shutil.which("llama-server")
    if found:
        return found
    raise RuntimeError("no llama-server binary -- call POST /runtime/build first")


def _has_cuda() -> bool:
    return shutil.which("nvidia-smi") is not None and shutil.which("nvcc") is not None


async def _run_step(stage: str, *cmd: str, cwd: Path | None = None) -> None:
    """One build step, its own subprocess, output tailed into _build_state
    so a stuck or failing step is visible rather than a silent multi-minute
    wait. Raises on nonzero exit -- the caller decides what that means for
    _build_state.error."""
    _build_state["stage"] = stage
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        if line:
            _build_state["log_tail"].append(line)
            del _build_state["log_tail"][:-_LOG_TAIL_MAX]
    code = await proc.wait()
    if code != 0:
        raise RuntimeError(f"{stage} failed (exit {code}): {cmd[0]} {' '.join(cmd[1:3])}...")


async def _run_build() -> None:
    src_dir = Path(config.AGENT_RUNTIME_DIR) / "src" / "llama.cpp"
    build_dir = src_dir / "build"
    try:
        if not (src_dir / ".git").exists():
            src_dir.parent.mkdir(parents=True, exist_ok=True)
            await _run_step("cloning", "git", "clone", "--depth", "1",
                             LLAMA_CPP_REPO, str(src_dir))

        cuda = _has_cuda()
        # -DBUILD_SHARED_LIBS=OFF: without it llama-server links against
        # libggml-cuda.so/etc sitting in build_dir/bin -- fine as long as
        # that build tree exists, but binary_path() only ever copies the
        # one executable out of it (below), so deleting the checkout later
        # to reclaim disk breaks a binary that looked complete. Not native
        # vs GGML_NATIVE, though: this build is compiled ON the machine
        # it'll run on, so targeting its exact GPU (CMAKE_CUDA_ARCHITECTURES
        # below) is correct here, unlike ci.yml's bundled build which has
        # to guess at an arbitrary downloader's CPU instead.
        cmake_args = ["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
                      "-DBUILD_SHARED_LIBS=OFF"]
        # native, not a hardcoded compute-capability list -- the
        # one number this session had to pass by hand (`75`, for the GTX
        # 1650 actually on this laptop) is exactly what "native" replaces:
        # detected from whatever GPU the build machine actually has.
        if cuda:
            cmake_args += ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=native"]
        await _run_step("configuring", *cmake_args, cwd=src_dir)

        import os
        jobs = str(os.cpu_count() or 4)
        await _run_step("building", "cmake", "--build", str(build_dir),
                         "--target", "llama-server", "-j", jobs, cwd=src_dir)

        _build_state["stage"] = "installing"
        dest = binary_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        built = build_dir / "bin" / "llama-server"
        if not built.exists():
            raise RuntimeError(f"build reported success but {built} is missing")
        shutil.copy2(built, dest)
        dest.chmod(0o755)
        _build_state["done"] = True
    except Exception as exc:  # subprocess failure, missing tool, disk full -- never a hang
        _build_state["error"] = str(exc)
    finally:
        _build_state["active"] = False


def start_build() -> bool:
    """Returns False without starting anything if a build is already active
    or the binary already exists -- callers check is_available() first for
    the "nothing to do" case; this only guards concurrent builds."""
    if _build_state["active"]:
        return False
    _build_state.update(active=True, stage="queued", log_tail=[], done=False, error=None)
    asyncio.create_task(_run_build())
    return True


# ---- running the backend(s) ---------------------------------------------

# Two llama-server processes, chat and embed, same reasoning as
# README.md's "Running the backends": one process holds exactly one
# model, so a chat model and an embedding model need separate processes on
# separate ports regardless of who starts them.
_processes: dict[str, asyncio.subprocess.Process] = {}


def backend_running(name: str) -> bool:
    proc = _processes.get(name)
    return proc is not None and proc.returncode is None


async def start_chat_backend(model_id: str, port: int, ngl: int = 99) -> None:
    if backend_running("chat"):
        return
    _processes["chat"] = await asyncio.create_subprocess_exec(
        resolve_binary(), "-hf", model_id, "--port", str(port),
        "--jinja", "-ngl", str(ngl),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,  # survive a Ctrl-C to *this* process -- see stop_all_backends
    )


async def start_embed_backend(model_id: str, port: int) -> None:
    if backend_running("embed"):
        return
    # -ngl 0, deliberately: docs/SCOPE.md §6.4's measured finding, running
    # both models on the GPU at once crashed both processes on this exact
    # card. Confirmed again live during this session's own testing.
    _processes["embed"] = await asyncio.create_subprocess_exec(
        resolve_binary(), "-hf", model_id, "--port", str(port),
        "--embeddings", "--pooling", "mean", "-ngl", "0",
        "-ub", "2048", "-b", "2048",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )


async def stop_backend(name: str) -> None:
    proc = _processes.get(name)
    if proc is None or proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
    del _processes[name]


async def stop_all_backends() -> None:
    for name in list(_processes):
        await stop_backend(name)
