# Changelog

All notable changes to claude-rag are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.4.1] — 2026-10-10

### Changed
- README rewritten and reorganised (quick start, tools, how it works, embedding model choice, hooks, one configuration section, evaluation), with an Italian version (`README.it.md`)
- New architecture diagram as SVG in English and Italian (`docs/architecture.svg`, `docs/architecture.it.svg`), readable in light and dark theme; it replaces `docs/diagram.png` and `docs/diagram.html`

---

## [1.4.0] — 2026-10-09

### Added
- **Optional EmbeddingGemma 2 embedder** (`KB_EMBED_MODEL=embeddinggemma-2`, `KB_EMBED_DIM`, `KB_EMBED_THREADS`, `requirements-gemma.txt`): text-only, fp32 on CPU, with the model's query/document prefixes. Semantic search on the author's gold set: correct section in top 5 **0.71 → 0.92**, MRR 0.65 → 0.80; hybrid + rerank rank 1 0.77 → 0.79
- `kb_meta` table records the embedding model; changing model forces a full reindex instead of mixing incompatible vectors
- `eval_retrieval.py`: the semantic retriever uses the server's embedder; `KB_EVAL_GOLD` overrides the gold set path

### Changed
- MiniLM remains the default: nothing changes without `KB_EMBED_MODEL`

---

## [1.3.0] — 2026-09-24

### Added
- **Local reranker** (`reranker.py`): the top 10 hybrid candidates are rescored by a multilingual cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, Apache-2.0, ONNX int8, ~0.4 s on CPU). Correct section at rank 1 **0.64 → 0.80**, in top 5 0.90 → 0.94
- **Calibrated confidence**: Platt scaling maps the top rerank score to the probability that the answer is among the results (5-fold CV: accuracy 0.95, ECE 0.07). Below `KB_MIN_CONFIDENCE` (0.5) `kb_search` warns that the answer is probably not in the KB
- `KB_RERANK`, `KB_RERANK_CANDIDATES`, `KB_RERANK_THREADS`, `KB_RERANK_CACHE`, `KB_MIN_CONFIDENCE`
- `eval_retrieval.py`: `--rerank=<models>`, `--calibrate`, rerank latency, `server+rr` check of the real integration

### Changed
- `requirements.txt` lists what the server actually imports (`fastembed`, not `sentence-transformers`); `scikit-learn` only for `--calibrate`
- Search falls back to hybrid if the reranker model cannot be loaded

---

## [1.2.0] — 2026-09-24

### Changed
- **Hybrid search**: `kb_search()` now fuses BM25 (keyword, inverted index) and semantic cosine via Reciprocal Rank Fusion. Measured on a private gold set (50 answerable + 7 unanswerable queries): correct section in top 5 **0.74 → 0.90**, in top 10 **0.82 → 1.00**. Identifier-heavy queries (table names, ports, hostnames) gain most; natural-language queries stay at 0.90.
- Output shows `rrf`, `sem` and `bm25` per chunk
- `KB_SEARCH_MODE=semantic` restores the previous behaviour

### Added
- `eval/`: `eval_retrieval.py` + `gold_set.example.json` (the real gold set and results stay local, gitignored, since they contain KB content) comparing semantic, hook lexical, BM25, hybrid, server and an optional local cross-encoder reranker; never writes to `search_stats`
- `rank()`: side-effect-free ranking used by `search()` and by the eval

### Docs
- README, `architecture.html`, `docs/diagram.html` + `diagram.png`, `kb_rag_slides.html` describe the hybrid search; embedding model name corrected to `paraphrase-multilingual-MiniLM-L12-v2`

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
