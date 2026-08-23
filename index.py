"""
The retrieval index: every chunk of the corpus, with its vector.

Deliberately not a vector database. At 133 chunks a similarity search is a
loop over a list, and every small-corpus project surveyed reaches for Qdrant
or FAISS only once PDFs or tens of thousands of chunks arrive. docs/SCOPE.md
§6.5 called this in advance; this is that decision, implemented.

The cache is keyed by the *content* of each chunk, not by file mtime -- a
git checkout rewrites mtimes without changing a word, and editing one heading
in STATUS.md should not cost a full re-embed of the other ten documents.
"""

import hashlib
import json
from pathlib import Path

import config
import embeddings
from corpus import Chunk


def _chunk_key(chunk: Chunk) -> str:
    digest = hashlib.sha256()
    for part in (chunk.path, chunk.heading, chunk.text):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


# Bumped whenever anything that changes the meaning of a stored vector
# changes: the embedding model, the prefixes, the file format itself.
_CACHE_VERSION = 1


def _cache_signature() -> str:
    return "|".join(
        [
            str(_CACHE_VERSION),
            config.BACKEND,
            config.OLLAMA_EMBED_MODEL,
            config.EMBED_DOCUMENT_PREFIX,
            config.EMBED_QUERY_PREFIX,
        ]
    )


class Index:
    """Chunks plus their vectors, and the search over them."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self._position: dict[int, int] = {}
        self.ready = False
        self.error: str | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- persistence ----------------------------------------------------

    def _load_cached_vectors(self) -> dict[str, list[float]]:
        path = Path(config.AGENT_INDEX_PATH)
        if not path.exists():
            return {}
        try:
            blob = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}  # a corrupt cache is a slow start, never a crash
        if blob.get("signature") != _cache_signature():
            return {}
        return blob.get("vectors", {})

    def _save(self, by_key: dict[str, list[float]]) -> None:
        path = Path(config.AGENT_INDEX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"signature": _cache_signature(), "vectors": by_key}
        # Write-then-rename, same rule as the model downloads: never leave a
        # half-written cache that would read back as valid.
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(payload))
        partial.replace(path)

    # ---- building -------------------------------------------------------

    async def build(self, chunks: list[Chunk]) -> None:
        """Embed whatever the cache doesn't already have, then keep both."""
        self.chunks = chunks
        self.error = None
        cached = self._load_cached_vectors()

        keys = [_chunk_key(c) for c in chunks]
        missing = [(k, c) for k, c in zip(keys, chunks) if k not in cached]

        if missing:
            try:
                fresh = await embeddings.embed_documents([c.text for _, c in missing])
            except embeddings.EmbeddingError as exc:
                self.ready = False
                self.error = str(exc)
                return
            for (key, _), vector in zip(missing, fresh):
                cached[key] = vector

        self.vectors = [cached[k] for k in keys]
        # By identity, not equality: two chunks can hold the same text, and a
        # caller holding one of them means that one.
        self._position = {id(c): i for i, c in enumerate(chunks)}
        # Drop vectors for chunks that no longer exist, so an edited corpus
        # doesn't grow the cache forever.
        self._save({k: cached[k] for k in keys})
        self.ready = True

    def vector_of(self, chunk: Chunk) -> list[float]:
        """The stored vector for a chunk this index returned. Never re-embeds."""
        return self.vectors[self._position[id(chunk)]]

    # ---- searching ------------------------------------------------------

    async def search(self, question: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        if not self.ready:
            raise embeddings.EmbeddingError(self.error or "index is not built")
        k = top_k or config.RETRIEVAL_TOP_K
        query = await embeddings.embed_query(question)
        scored = [
            (chunk, embeddings.cosine(query, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
