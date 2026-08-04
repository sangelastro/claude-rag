# claude-rag — MCP RAG Server for Markdown Knowledge Bases

MCP server that indexes a folder of `.md` files into a local SQLite vector store and exposes semantic search as Claude Code tools.

**Stack**: Python · sentence-transformers (`all-MiniLM-L6-v2`, ~90MB, CPU-only) · SQLite · MCP stdio

## Tools exposed

| Tool | Description |
|---|---|
| `kb_search(query, top_k=5)` | Semantic search — returns top-K chunks with source file and section |
| `kb_reindex(force=False)` | Re-indexes files modified since last run (mtime-based) |
| `kb_stats()` | Shows indexed files, chunk counts, last update timestamps |

## Setup

### 1. Clone

```bash
# Default layout: repo sits inside the KB folder
# KB files (.md) go in the parent directory
git clone https://github.com/sangelastro/claude-rag ~/.claude/my-kb/rag
```

Or clone anywhere and point to your KB folder via env var (see step 3).

### 2. Install dependencies

```bash
cd ~/.claude/my-kb/rag
pip install -r requirements.txt
```

On first run the model (`all-MiniLM-L6-v2`, ~90MB) is downloaded automatically from HuggingFace.

### 3. Register in Claude Code

Add to `~/.claude.json` under `mcpServers`:

```json
"my-kb": {
  "command": "python",
  "args": ["/absolute/path/to/rag/server.py"],
  "env": {
    "KB_RAG_DIR": "/absolute/path/to/your/kb/folder"
  }
}
```

- **`KB_RAG_DIR`** — folder containing your `.md` files (default: `../` relative to `server.py`)
- **`KB_RAG_DB`** — SQLite database path (default: `kb.db` next to `server.py`)

If the repo is cloned inside the KB folder (as in the example above), both env vars can be omitted.

### 4. Restart Claude Code

The server starts automatically. On first launch it indexes all `.md` files in `KB_RAG_DIR`.

## Optional: Claude Code Hooks

Two hooks auto-inject KB context without explicit `kb_search` calls.
Approach inspired by [agd-memory](https://github.com/Pinperepette/agd-memory) (MIT).

| Hook | Script | What it does |
|---|---|---|
| `SessionStart` | `hooks/kb_session_start.py` | Injects KB table of contents at session start |
| `UserPromptSubmit` | `hooks/kb_recall.py` | Auto-injects top matching chunks before each prompt (keyword scoring, ~10ms, no model load) |

### Setup hooks

Copy the relevant sections from `hooks/hooks_example.json` into your `~/.claude/settings.json`, replacing the placeholder paths:

```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "*",
      "hooks": [{"type": "command", "command": "python /absolute/path/to/rag/hooks/kb_session_start.py"}]
    }],
    "UserPromptSubmit": [{
      "matcher": "*",
      "hooks": [{"type": "command", "command": "python /absolute/path/to/rag/hooks/kb_recall.py", "timeout": 5}]
    }]
  }
}
```

`kb_chunks.json` (used by `kb_recall.py`) is auto-generated next to `kb.db` on every reindex. No model loading in hooks — scoring uses token overlap only.

Hook behaviour is tunable via env vars:

| Variable | Default | Description |
|---|---|---|
| `KB_RAG_HOOK_TOP_K` | `3` | Max chunks injected per prompt |
| `KB_RAG_HOOK_MIN_SCORE` | `0.15` | Minimum score to trigger injection |
| `KB_RAG_HOOK_MIN_WORDS` | `4` | Skip prompts shorter than N words |
| `KB_RAG_HOOK_TOKEN_BUDGET` | `6000` | Max chars injected (~4 chars/token) |

## File structure

```
rag/
├── server.py           # MCP server
├── requirements.txt
├── .gitignore
├── README.md
├── hooks/
│   ├── kb_session_start.py   # SessionStart hook
│   ├── kb_recall.py          # UserPromptSubmit hook
│   └── hooks_example.json    # Hook config template
├── architecture.html   # Technical documentation
└── kb_rag_slides.html  # Architecture slide deck
```

`kb.db` and `kb_chunks.json` are generated locally and excluded from git.

## How it works

1. **Chunking** — each `.md` file is split on `##` headers; frontmatter is stripped
2. **Embedding** — chunks are encoded with `all-MiniLM-L6-v2` (384 dimensions)
3. **Storage** — vectors stored as `float32` BLOBs in SQLite + `kb_chunks.json` for hooks
4. **Search** — cosine similarity computed in numpy over all chunks; top-K returned
5. **Hooks** — keyword scoring on `kb_chunks.json` (no model), injected before each prompt
6. **Invalidation** — mtime-based: only modified files are re-indexed on startup

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `KB_RAG_DIR` | `../` (relative to `server.py`) | Folder with `.md` files to index |
| `KB_RAG_DB` | `./kb.db` (next to `server.py`) | SQLite database path |
| `KB_RAG_NAME` | `kb-rag` | MCP server name |

## Credits

Hook architecture inspired by [agd-memory](https://github.com/Pinperepette/agd-memory) by Pinperepette (MIT License) — in particular the UserPromptSubmit recall pattern and guard rail logic.
