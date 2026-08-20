import os

# "ollama" or "llamacpp". Ollama can list and switch between many pulled
# models from one running process; llama-server loads exactly one model per
# process, so under llamacpp "model selection" means picking which server to
# talk to, not picking from a list one server offers. See docs/SCOPE.md §6.4.
BACKEND = os.environ.get("AGENT_BACKEND", "llamacpp")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLAMACPP_HOST = os.environ.get("LLAMACPP_HOST", "http://localhost:8080")

AGENT_SERVER_PORT = int(os.environ.get("AGENT_SERVER_PORT", "8420"))

BACKEND_TIMEOUT_SECONDS = 3.0
