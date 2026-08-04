# Credits

## agd-memory — Hook Architecture

The Claude Code hook system in this project (`hooks/kb_session_start.py` and
`hooks/kb_recall.py`) is directly inspired by
**[agd-memory](https://github.com/Pinperepette/agd-memory)** by
[@Pinperepette](https://github.com/Pinperepette), released under the
[MIT License](https://github.com/Pinperepette/agd-memory/blob/main/LICENSE).

Specifically adapted from agd-memory:

| Concept | agd-memory | This project |
|---|---|---|
| `SessionStart` hook | `hooks/agd-memory-bootstrap.sh` — injects AGD memory TOC | `hooks/kb_session_start.py` — injects SQLite chunk TOC |
| `UserPromptSubmit` hook | `hooks/agd-memory-recall.py` — keyword scoring + cognate matching | `hooks/kb_recall.py` — token overlap / IDF scoring on `kb_chunks.json` |
| Guard rails | min words, code-paste detection, score threshold, ratio filter | Same pattern, adapted for this project's scoring function |
| Token budget | `AGD_RECALL_TOKEN_BUDGET` env var | `KB_RAG_HOOK_TOKEN_BUDGET` env var |

**Key difference**: agd-memory uses keyword overlap for retrieval (no model
required at hook time). This project uses the same approach for the hooks
(fast, ~10ms, no model load), while the MCP tool `kb_search` uses full
semantic search via `sentence-transformers` cosine similarity.
