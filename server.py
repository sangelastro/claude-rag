"""
RAG MCP Server
Indexes .md files by ## section and answers semantic queries.
Auto-reindexes on startup when files have been modified.
"""
import json
import os
import sys
import sqlite3

os.environ.pop("SSL_CERT_FILE", None)
from pathlib import Path
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from mcp.server.mcpserver.server import MCPServer

KB_DIR = Path(os.environ.get("KB_RAG_DIR", str(Path(__file__).parent.parent)))
DB_PATH = Path(os.environ.get("KB_RAG_DB", str(Path(__file__).parent / "kb.db")))
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── DB ────────────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id        INTEGER PRIMARY KEY,
            file      TEXT NOT NULL,
            section   TEXT NOT NULL,
            content   TEXT NOT NULL,
            embedding BLOB NOT NULL,
            mtime     REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file ON chunks(file)")
    conn.commit()
    return conn


# ── CHUNKING ──────────────────────────────────────────────────────────────────

def chunk_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Rimuovi frontmatter YAML
    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                start = i + 1
                break

    file_title = ""
    chunks = []
    current_section = ""
    current_lines = []

    def flush(section, body_lines):
        body = "\n".join(body_lines).strip()
        if body:
            header = f"[{path.stem}] {file_title}"
            if section:
                header += f" > {section}"
            chunks.append({
                "file": path.name,
                "section": section or file_title or path.stem,
                "content": f"{header}\n\n{body}",
            })

    for line in lines[start:]:
        if line.startswith("# ") and not line.startswith("## "):
            file_title = line[2:].strip()
            continue
        if line.startswith("## "):
            flush(current_section, current_lines)
            current_section = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush(current_section, current_lines)
    return chunks


# ── INDEXING ──────────────────────────────────────────────────────────────────

def needs_reindex(conn: sqlite3.Connection, path: Path) -> bool:
    mtime = path.stat().st_mtime
    row = conn.execute(
        "SELECT mtime FROM chunks WHERE file = ? LIMIT 1", (path.name,)
    ).fetchone()
    return row is None or row[0] < mtime


def index_file(conn: sqlite3.Connection, path: Path) -> int:
    model = get_model()
    mtime = path.stat().st_mtime
    chunks = chunk_file(path)
    if not chunks:
        return 0

    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    conn.execute("DELETE FROM chunks WHERE file = ?", (path.name,))
    conn.executemany(
        "INSERT INTO chunks (file, section, content, embedding, mtime) VALUES (?,?,?,?,?)",
        [
            (c["file"], c["section"], c["content"],
             emb.astype(np.float32).tobytes(), mtime)
            for c, emb in zip(chunks, embeddings)
        ],
    )
    conn.commit()
    return len(chunks)


def export_chunks_json(conn: sqlite3.Connection) -> None:
    """Export chunk texts to JSON for the UserPromptSubmit hook (no model needed)."""
    rows = conn.execute("SELECT file, section, content FROM chunks").fetchall()
    chunks = [{"file": r[0], "section": r[1], "content": r[2]} for r in rows]
    json_path = DB_PATH.parent / "kb_chunks.json"
    json_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=None), encoding="utf-8")


def reindex_all(conn: sqlite3.Connection, force: bool = False) -> tuple[int, int]:
    md_files = [p for p in KB_DIR.glob("*.md") if p.name != "MEMORY.md"]
    updated_files = 0
    total_chunks = 0

    for path in md_files:
        if force or needs_reindex(conn, path):
            n = index_file(conn, path)
            total_chunks += n
            updated_files += 1

    # Rimuovi chunk di file cancellati
    existing = {p.name for p in md_files}
    indexed = {r[0] for r in conn.execute("SELECT DISTINCT file FROM chunks").fetchall()}
    for orphan in indexed - existing:
        conn.execute("DELETE FROM chunks WHERE file = ?", (orphan,))
    conn.commit()

    export_chunks_json(conn)
    return updated_files, total_chunks


# ── SEARCH ────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def search(query: str, top_k: int = 5) -> list[dict]:
    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].astype(np.float32)

    conn = get_db()
    rows = conn.execute("SELECT file, section, content, embedding FROM chunks").fetchall()
    conn.close()

    scored = []
    for file, section, content, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        score = cosine_sim(q_emb, emb)
        scored.append({"file": file, "section": section, "content": content, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_results(results: list[dict]) -> str:
    if not results:
        return "Nessun risultato trovato nella KB."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"--- [{r['file']} > {r['section']}] (score: {r['score']:.3f}) ---\n{r['content']}"
        )
    return "\n\n".join(parts)


# ── MCP SERVER (FastMCP) ──────────────────────────────────────────────────────

_server_name = os.environ.get("KB_RAG_NAME", "kb-rag")
mcp = MCPServer(_server_name)


@mcp.tool()
def kb_search(query: str, top_k: int = 5) -> str:
    """
    Search the knowledge base using semantic similarity.
    Returns the most relevant chunks with source file and section.

    Args:
        query: Search query (e.g. 'database connection string', 'API authentication')
        top_k: Maximum number of results (default 5, max 10)
    """
    results = search(query, min(top_k, 10))
    return format_results(results)


@mcp.tool()
def kb_reindex(force: bool = False) -> str:
    """
    Reindicizza la KB dopo aver aggiornato i file .md.
    Chiamare dopo ogni modifica ai file della knowledge base.

    Args:
        force: Se True, reindicizza tutti i file anche se non modificati
    """
    conn = get_db()
    updated, chunks = reindex_all(conn, force=force)
    conn.close()
    return f"Reindex completato: {updated} file aggiornati, {chunks} chunk indicizzati."


@mcp.tool()
def kb_stats() -> str:
    """Mostra statistiche sull'indice: file indicizzati, chunk totali, data ultimo aggiornamento."""
    conn = get_db()
    rows = conn.execute(
        "SELECT file, COUNT(*) as n, MAX(mtime) as t FROM chunks GROUP BY file ORDER BY file"
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()

    lines = [f"Totale chunk: {total}\n"]
    for file, n, mtime in rows:
        ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  {file}: {n} chunk (aggiornato {ts})")
    return "\n".join(lines)


# ── STARTUP ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("KB RAG: caricamento modello...", file=sys.stderr)
    get_model()  # scarica modello al primo avvio (~80MB)

    conn = get_db()
    updated, chunks = reindex_all(conn)
    conn.close()

    if updated:
        print(f"KB RAG: reindicizzati {updated} file ({chunks} chunk)", file=sys.stderr)
    else:
        print("KB RAG: indice aggiornato, nessun file modificato", file=sys.stderr)

    print("KB RAG: server pronto", file=sys.stderr)
    mcp.run()
