"""
The corpus, split into citable pieces.

A2b (docs/SCOPE.md §6.5) replaces A2a's paste-the-whole-corpus prompt with
retrieval, and the unit of retrieval has to be the unit of citation: §7 makes
citing a hard requirement, so a chunk that spans two sections gives the model
nothing honest to cite. Markdown headings are already that unit here -- the
eleven reachable docs carry 135 of them, roughly 730 bytes apiece -- so the
split follows the document's own structure rather than a token count.

No dependency for this: header-aware splitting of eleven files is thirty
lines, and langchain/llama_index would each pull a tree of packages onto a
laptop whose whole point (§6.4) is running on less.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import config

# Fenced code blocks contain lines that look like headings ("# comment"),
# and splitting inside a fence would produce a chunk with an unclosed fence.
_FENCE_RE = re.compile(r"^(```|~~~)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# A section longer than this is split further. Chosen against the corpus:
# only a handful of sections exceed it, and 2000 characters is ~500 tokens,
# so a top-k of 6 still leaves room for the question and the answer inside a
# 4096-token context -- the floor a GTX 1650 tier actually gets (§6.4).
MAX_CHUNK_CHARS = 2000
OVERLAP_CHARS = 200


@dataclass(frozen=True)
class Chunk:
    """One citable piece: where it came from, and what it says."""

    path: str  # repo-relative, e.g. "docs/STATUS.md"
    heading: str  # "STATUS > Bugs found and fixed > 1. ..." -- the full trail
    text: str

    @property
    def citation(self) -> str:
        """What the model is asked to write, and what the UI links back to."""
        return self.path if not self.heading else f"{self.path} § {self.heading}"


def _heading_trail(stack: list[tuple[int, str]]) -> str:
    """The trail, minus the document's own H1 title.

    Every one of these files opens with an H1 restating what the file is, so
    keeping it would put "PdM Maestro — orientation for whoever picks this up
    next" in front of all nine CLAUDE.md citations, next to the filename that
    already says it. It only survives when it is the whole trail.
    """
    titles = [title for _, title in stack]
    if len(titles) > 1 and stack[0][0] == 1:
        titles = titles[1:]
    return " > ".join(titles)


def split_markdown(path: str, text: str) -> list[Chunk]:
    """Split one document at its headings, deepest heading wins the chunk."""
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []  # (level, title), outermost first
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        content = "\n".join(body).strip()
        body.clear()
        if not content:
            return
        trail = _heading_trail(stack)
        pieces = _split_long(content)
        for i, piece in enumerate(pieces, start=1):
            # A section that had to be split would otherwise produce two
            # chunks with byte-identical citations, and a reader following
            # one of them has no way to tell which half was meant.
            heading = trail if len(pieces) == 1 else f"{trail} (part {i} of {len(pieces)})"
            chunks.append(Chunk(path=path, heading=heading, text=piece))

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING_RE.match(line)
        if match is None:
            body.append(line)
            continue
        flush()
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2).strip()))
    flush()
    return chunks


def _split_long(text: str) -> list[str]:
    """Overlapping character windows, cut at a line break where one is near.

    Only long sections reach this. The overlap exists so a fact stated across
    the seam is whole in at least one window.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        if end < len(text):
            newline = text.rfind("\n", start + MAX_CHUNK_CHARS // 2, end)
            if newline != -1:
                end = newline
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return [p for p in pieces if p]


def load_chunks(paths: list[str]) -> list[Chunk]:
    """Read the reachable corpus and split every document in it."""
    root = Path(config.AGENT_MAESTRO_ROOT)
    chunks: list[Chunk] = []
    for rel in paths:
        chunks.extend(split_markdown(rel, (root / rel).read_text()))
    return chunks
