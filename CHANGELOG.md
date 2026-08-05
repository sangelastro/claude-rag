# Changelog

All notable changes to claude-rag are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026-08-05

### Performance
- **In-memory embedding cache**: all chunk embeddings loaded into a numpy matrix at startup — eliminates full table scan on every `kb_search()` call
- **Vectorized cosine similarity**: single matrix multiplication (`@`) replaces Python loop over 1199 chunks (~50× faster per query)
- **ONNX thread limiting**: `OMP_NUM_THREADS=2`, `ORT_NUM_INTRA_OP_THREADS=2` — prevents CPU saturation on multi-core systems
- Cache automatically invalidated after `kb_reindex()` and reloaded lazily on next search

---

## [1.0.0] — 2026-08-05

### Added
- **Token savings tracking**: `search_stats` SQLite table records every search with chars served vs full-file baseline
- **`kb_savings()` tool**: cumulative token reduction stats, broken down by source (`mcp` / `hook`)
- Savings also tracked in `hooks/kb_recall.py` for automatic hook injections

### Changed
- Embedding model: `all-MiniLM-L6-v2` → `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, better Italian support)
- **Multi-chunk splitting**: sections longer than `CHUNK_MAX_CHARS` (400 chars) split into overlapping sub-chunks with `CHUNK_OVERLAP` (80 chars) — prevents silent truncation at model's 128-token limit
- SSL fix: `certifi` CA bundle replaces Windows cert store workaround
- `HF_HUB_OFFLINE=1` added to MCP env to skip HuggingFace network checks at startup

### New env vars
| Variable | Default | Description |
|---|---|---|
| `CHUNK_MAX_CHARS` | `400` | Max chars per chunk before splitting |
| `CHUNK_OVERLAP` | `80` | Overlap chars between adjacent sub-chunks |

---

## [0.2.0] — 2026-08-04

### Added
- Architecture diagram (`docs/diagram.png`, `docs/diagram.html`)
- `kb_session_start.py` hook: injects KB table of contents at session start
- `hooks_example.json`: ready-to-use hook configuration template

### Changed
- `UserPromptSubmit` hook scoring: IDF-weighted token overlap replaces simple overlap
- Anti-noise ratio filter: prunes results scoring < 30% of top result

---

## [0.1.0] — 2026-07-01

### Added
- Initial release
- `server.py`: MCP server with `kb_search()`, `kb_reindex()`, `kb_stats()` tools
- Chunking on `##` headers with frontmatter stripping
- SQLite vector store with `float32` BLOB embeddings
- `kb_recall.py` hook: keyword scoring on `kb_chunks.json` (~10ms, no model load)
- `export_chunks_json()`: auto-exports chunk texts after every reindex
- mtime-based incremental reindex (only modified files)
