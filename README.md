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

## File structure

```
rag/
├── server.py          # MCP server
├── requirements.txt
├── .gitignore
├── README.md
├── architecture.html  # Technical documentation
└── kb_rag_slides.html # Architecture slide deck
```

`kb.db` is generated locally and excluded from git.

## How it works

1. **Chunking** — each `.md` file is split on `##` headers; frontmatter is stripped
2. **Embedding** — chunks are encoded with `all-MiniLM-L6-v2` (384 dimensions)
3. **Storage** — vectors stored as `float32` BLOBs in SQLite (no external vector DB)
4. **Search** — cosine similarity computed in numpy over all chunks; top-K returned
5. **Invalidation** — mtime-based: only modified files are re-indexed on startup

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `KB_RAG_DIR` | `../` (relative to `server.py`) | Folder with `.md` files to index |
| `KB_RAG_DB` | `./kb.db` (next to `server.py`) | SQLite database path |
