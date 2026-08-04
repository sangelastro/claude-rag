#!/usr/bin/env python3
"""
UserPromptSubmit hook: auto-inject relevant KB chunks before each prompt.

Approach inspired by agd-memory's UserPromptSubmit recall hook:
  github.com/Pinperepette/agd-memory (MIT License)
Differences:
  - Uses a pre-exported kb_chunks.json (no model load, ~10ms)
  - Scoring: token overlap with IDF-like weighting instead of cognate matching
  - Guard rails and budget logic adapted from agd-memory's implementation

Configuration via env vars:
  KB_RAG_HOOK_TOP_K        max chunks to inject (default: 3)
  KB_RAG_HOOK_MIN_SCORE    minimum score for injection (default: 0.15)
  KB_RAG_HOOK_MIN_WORDS    skip prompts shorter than N words (default: 4)
  KB_RAG_HOOK_TOKEN_BUDGET max chars injected (~4 chars/token) (default: 6000)
"""
import json
import math
import os
import re
import sys
from pathlib import Path

CHUNKS_JSON = Path(__file__).parent.parent / "kb_chunks.json"

TOP_K = int(os.environ.get("KB_RAG_HOOK_TOP_K", "3"))
MIN_SCORE = float(os.environ.get("KB_RAG_HOOK_MIN_SCORE", "0.15"))
MIN_WORDS = int(os.environ.get("KB_RAG_HOOK_MIN_WORDS", "4"))
TOKEN_BUDGET = int(os.environ.get("KB_RAG_HOOK_TOKEN_BUDGET", "6000"))

STOPWORDS = {
    # Italian
    "il", "lo", "la", "i", "gli", "le", "un", "una", "uno",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "e", "o", "ma", "se", "che", "non", "del", "della", "dei",
    "come", "quando", "dove", "cosa", "chi", "quale", "questo", "quello",
    # English
    "the", "a", "an", "of", "to", "in", "for", "on", "with",
    "at", "by", "from", "is", "are", "was", "be", "or", "and",
    "not", "this", "it", "that", "as", "have", "has", "do", "does",
}


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r'\b\w{3,}\b', text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def score_chunk(query_tokens: set[str], chunk_content: str) -> float:
    """
    Token overlap score weighted by inverse chunk length (IDF-inspired).
    Adapted from agd-memory's overlap scoring logic.
    """
    chunk_tokens = tokenize(chunk_content)
    if not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / math.sqrt(len(chunk_tokens))


def is_code_paste(prompt: str) -> bool:
    stripped = prompt.strip()
    return stripped.startswith("```") or stripped.startswith("    ")


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    prompt = payload.get("prompt", "")

    # Guard rails (pattern from agd-memory)
    if len(prompt.split()) < MIN_WORDS:
        sys.exit(0)
    if is_code_paste(prompt):
        sys.exit(0)
    if not CHUNKS_JSON.exists():
        sys.exit(0)

    try:
        chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    except Exception:
        sys.exit(0)

    query_tokens = tokenize(prompt)
    if not query_tokens:
        sys.exit(0)

    scored = [
        (score_chunk(query_tokens, c["content"]), c)
        for c in chunks
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored or scored[0][0] < MIN_SCORE:
        sys.exit(0)

    # Anti-noise ratio filter (from agd-memory: prunes results far below top)
    top_score = scored[0][0]
    candidates = [
        (s, c) for s, c in scored[:TOP_K]
        if s >= top_score * 0.3
    ]

    out_parts = ["[KB auto-context]"]
    chars_used = 0
    for s, chunk in candidates:
        entry = f"\n--- [{chunk['file']} › {chunk['section']}] ---\n{chunk['content']}"
        if chars_used + len(entry) > TOKEN_BUDGET:
            break
        out_parts.append(entry)
        chars_used += len(entry)

    if len(out_parts) > 1:
        print("\n".join(out_parts))

    sys.exit(0)


if __name__ == "__main__":
    main()
