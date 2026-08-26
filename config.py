import os
import tomllib
from pathlib import Path

# A config file for "easy setup" (the setup wizard, setup.py, writes one) --
# without it every setting below still works exactly as before, from env
# vars and hardcoded defaults alone. Priority is env var > config file >
# hardcoded default, so a deployment env (systemd unit, docker) can always
# override a file underneath it without editing that file.
CONFIG_PATH = Path(os.environ.get(
    "AGENT_CONFIG_PATH", os.path.expanduser("~/.config/pdm_agent/config.toml")
))


def _load_file_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}  # a broken config file falls back to defaults, never a crash


_file = _load_file_config()


def _get(key: str, default: str) -> str:
    return os.environ.get(key, str(_file.get(key, default)))


# "ollama" or "llamacpp". Ollama can list and switch between many pulled
# models from one running process; llama-server loads exactly one model per
# process, so under llamacpp "model selection" means picking which server to
# talk to, not picking from a list one server offers. See docs/SCOPE.md §6.4.
BACKEND = _get("AGENT_BACKEND", "llamacpp")

OLLAMA_HOST = _get("OLLAMA_HOST", "http://localhost:11434")
LLAMACPP_HOST = _get("LLAMACPP_HOST", "http://localhost:8080")

# A2b needs embeddings as well as chat, and the two backends differ in a way
# that has to show up in configuration rather than being papered over:
#
#   * Ollama holds several models in one daemon, so one host serves both and
#     "which model" is a name in the request body. This is the concrete
#     advantage docs/SCOPE.md §6.4 claimed for it, now confirmed.
#   * llama-server loads exactly one model per process. Its router mode and
#     llama-swap both keep only one model resident and pay a 3-10s reload to
#     switch -- and a switch would land on every single question, since every
#     question embeds before it chats. So llamacpp means a *second* server on
#     a second port, not a second model on the same one.
LLAMACPP_EMBED_HOST = os.environ.get("LLAMACPP_EMBED_HOST", "http://localhost:8081")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# nomic-embed-text v1.5: 768 dimensions, 8192-token context, and the only
# small embedder with both a first-party Ollama pull and a clean GGUF for
# llama-server -- so the same index is reproducible on either backend
# (§6.4's "the models are meant to be interchangeable across machines").
#
# It is asymmetric: documents and queries get different prefixes, and getting
# them wrong degrades results silently rather than raising. They live here so
# there is exactly one place to be wrong.
EMBED_QUERY_PREFIX = "search_query: "
EMBED_DOCUMENT_PREFIX = "search_document: "

# Chunks pasted into the answer prompt. Small on purpose: the GTX 1650 floor
# realistically gets n_ctx 2048-4096, and a 3B model mis-cites more the more
# candidates it is shown.
RETRIEVAL_TOP_K = int(os.environ.get("AGENT_TOP_K", "5"))

EMBED_TIMEOUT_SECONDS = 60.0

# How far behind the best-matching source a cited source may fall before the
# citation is called unsupported. Provisional: calibrated against a six
# question sample, where the honest failure was the check accusing a correct
# one-sentence answer over a 0.039 gap. It wants a real eval set, and until it
# has one this number is a starting point, not a measurement.
CITATION_MARGIN = float(os.environ.get("AGENT_CITATION_MARGIN", "0.05"))

# Below this, the answer is too short for a similarity check to mean anything.
CITATION_MIN_ANSWER_CHARS = 40

# Embedding the corpus takes a minute or so; the result is cached next to the
# downloaded models rather than in the repo, same rule as the .gguf files.
AGENT_INDEX_PATH = os.environ.get(
    "AGENT_INDEX_PATH", os.path.expanduser("~/.cache/pdm_agent/index.json")
)

AGENT_SERVER_PORT = int(_get("AGENT_SERVER_PORT", "8420"))

BACKEND_TIMEOUT_SECONDS = 3.0

# Outside the repo by default, on purpose: a downloaded .gguf must never be
# able to land under apps/agent/ by accident. See docs/SCOPE.md §6.
AGENT_MODELS_DIR = os.environ.get(
    "AGENT_MODELS_DIR", os.path.expanduser("~/.cache/pdm_agent/models")
)

# Where a downloaded llama-server binary lands (server/runtime.py). Same
# "outside the repo, never bundled" rule as AGENT_MODELS_DIR, and the same
# reason: a fresh clone must never accidentally ship a platform-specific
# binary.
AGENT_RUNTIME_DIR = os.environ.get(
    "AGENT_RUNTIME_DIR", os.path.expanduser("~/.cache/pdm_agent/runtime")
)

# Curated, not fetched from HF's search API -- four known-good targets,
# spanning the hardware tiers in docs/SCOPE.md §6.4. Filenames and
# size_bytes are the real values for each repo's Q4_K_M file, resolved by
# hand against the HF API; keep them in sync if a repo re-quantizes.
# Lives here, not in main.py, so setup.py can list it without importing
# main (which builds the corpus and needs AGENT_MAESTRO_ROOT at import time).
CATALOG = [
    {
        "id": "qwen2.5-1.5b-instruct-q4",
        "label": "Qwen2.5 1.5B Instruct (Q4_K_M)",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_bytes": 1117320736,
        "quant": "Q4_K_M",
    },
    {
        "id": "qwen2.5-3b-instruct-q4",
        "label": "Qwen2.5 3B Instruct (Q4_K_M)",
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_bytes": 2104932768,
        "quant": "Q4_K_M",
    },
    {
        "id": "qwen2.5-7b-instruct-q4",
        "label": "Qwen2.5 7B Instruct (Q4_K_M)",
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_bytes": 4683074240,
        "quant": "Q4_K_M",
    },
    {
        "id": "llama-3.2-3b-instruct-q4",
        "label": "Llama 3.2 3B Instruct (Q4_K_M)",
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_bytes": 2019377696,
        "quant": "Q4_K_M",
    },
]

# The one embedding model this server knows how to run (see the EMBED_QUERY_
# PREFIX comment above for why nomic-embed-text specifically). Named here,
# not just inlined in main.py's BackendStartRequest default, so setup.py's
# wizard can start the same backend it advertises without a second copy of
# the string to keep in sync.
EMBED_MODEL_HF_ID = "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"

# This server used to assume it was checked out three directories under a
# Maestro tree (apps/agent/server/config.py -> ... -> repo root) and derived
# this from __file__. As its own repository, cloned anywhere, that guess is
# simply wrong on everyone's machine, so there is no default left: this now
# has to be set, by the setup wizard (setup.py) writing it to the config
# file, or by hand.
# A missing/wrong value fails loudly the first time a corpus path is read
# (main.py), not silently here.
AGENT_MAESTRO_ROOT = _get("AGENT_MAESTRO_ROOT", "")

CHAT_TIMEOUT_SECONDS = 120.0
