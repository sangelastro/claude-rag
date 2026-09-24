# Copyright (c) 2026 Sergio Angelastro — MIT License
"""
RAG MCP Server
Indexes .md files by ## section and answers semantic queries.
Auto-reindexes on startup when files have been modified.
"""
import json
import math
import os
import re
import sys
import sqlite3
from collections import Counter, defaultdict

import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"]      = certifi.where()
# limita i thread ONNX Runtime per evitare spike CPU che congelano il PC
os.environ.setdefault("OMP_NUM_THREADS",        "2")
os.environ.setdefault("ONNX_NUM_THREADS",       "2")
os.environ.setdefault("ORT_NUM_INTRA_OP_THREADS","2")
os.environ.setdefault("ORT_NUM_INTER_OP_THREADS","1")

from pathlib import Path
from datetime import datetime

import numpy as np
from fastembed import TextEmbedding
from mcp.server.mcpserver.server import MCPServer

KB_DIR     = Path(os.environ.get("KB_RAG_DIR", str(Path(__file__).parent.parent)))
DB_PATH    = Path(os.environ.get("KB_RAG_DB",  str(Path(__file__).parent / "kb.db")))
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# max token del modello = 128 → ~400 chars per chunk (stima conservativa per italiano)
# overlap = 80 chars per preservare contesto tra chunk adiacenti
CHUNK_MAX_CHARS  = int(os.environ.get("CHUNK_MAX_CHARS",  "400"))
CHUNK_OVERLAP    = int(os.environ.get("CHUNK_OVERLAP",     "80"))

# hybrid = BM25 + semantico fusi con RRF (default); semantic = solo coseno (comportamento <= 1.1.0)
# Scelto su eval/gold_set.json: sec@5 0.74 → 0.90, sec@10 0.82 → 1.00
SEARCH_MODE = os.environ.get("KB_SEARCH_MODE", "hybrid")
FUSION_DEPTH = 50   # candidati presi da ciascun ranking prima della fusione
RRF_K        = 60   # costante standard di Reciprocal Rank Fusion
BM25_K1, BM25_B = 1.2, 0.75

# Secondo stadio: cross-encoder locale che riordina i candidati ibridi (vedi reranker.py).
# KB_RERANK=off lo disattiva; se il modello non si carica si resta sull'ibrida, senza errori.
RERANK_MODEL      = os.environ.get("KB_RERANK", "mmarco-mminilm")
RERANK_CANDIDATES = int(os.environ.get("KB_RERANK_CANDIDATES", "10"))   # l'ibrida ha gia' la risposta nei primi 10
RERANK_THREADS    = int(os.environ.get("KB_RERANK_THREADS", "4"))
# sotto questa confidenza kb_search avvisa che la risposta probabilmente non e' nella KB
MIN_CONFIDENCE    = float(os.environ.get("KB_MIN_CONFIDENCE", "0.5"))

_model: TextEmbedding = None

# cache in-memory degli embedding: caricati una volta all'avvio/reindex
_emb_matrix: np.ndarray = None   # shape (N, 384)
_emb_meta: list = None           # lista di {file, section, content}
_bm25: "BM25Index" = None        # indice lessicale sugli stessi chunk
_reranker = None                 # None = non ancora caricato, False = non disponibile


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def get_reranker():
    global _reranker
    if _reranker is None:
        if RERANK_MODEL == "off":
            _reranker = False
        else:
            try:
                from reranker import Reranker
                _reranker = Reranker(RERANK_MODEL, threads=RERANK_THREADS)
            except Exception as e:
                print(f"KB RAG: reranker non disponibile ({e}), resto sulla ricerca ibrida", file=sys.stderr)
                _reranker = False
    return _reranker


class BM25Index:
    """BM25 su indice invertito. Cattura gli identificatori esatti (nomi di tabella, porte, hostname)
    che l'embedding MiniLM a 128 token non distingue."""

    def __init__(self, texts: list[str]):
        self.postings = defaultdict(list)   # term -> [(doc, tf)]
        lengths = []
        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.n = len(texts)
        self.lengths = np.array(lengths, dtype=np.float32)
        avg = float(self.lengths.mean()) if self.n else 1.0
        self.norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths / avg)
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w{2,}", text.lower())

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in set(self.tokenize(query)):
            for doc, tf in self.postings.get(term, ()):
                s[doc] += self.idf[term] * tf * (BM25_K1 + 1) / (tf + self.norm[doc])
        return s


def _load_emb_cache(conn: sqlite3.Connection) -> None:
    """Carica embedding e indice BM25 in RAM. Chiamato all'avvio e dopo reindex."""
    global _emb_matrix, _emb_meta, _bm25
    rows = conn.execute("SELECT file, section, content, embedding FROM chunks").fetchall()
    if not rows:
        _emb_matrix = np.zeros((0, 384), dtype=np.float32)
        _emb_meta = []
        _bm25 = BM25Index([])
        return
    _emb_meta = [{"file": r[0], "section": r[1], "content": r[2]} for r in rows]
    _emb_matrix = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    _bm25 = BM25Index([m["content"] for m in _emb_meta])
    get_reranker()   # precarica qui (avvio / reindex) invece che alla prima ricerca: ~2 s


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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_stats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            source      TEXT NOT NULL,
            query       TEXT NOT NULL,
            top_k       INTEGER,
            chunks_ret  INTEGER,
            chars_ret   INTEGER,
            files_hit   INTEGER,
            chars_full  INTEGER,
            chars_saved INTEGER
        )
    """)
    conn.commit()
    return conn


def _record_stat(conn: sqlite3.Connection, source: str, query: str,
                 top_k: int, results: list[dict]) -> None:
    if not results:
        return
    chars_ret = sum(len(r["content"]) for r in results)
    files_hit = list({r["file"] for r in results})
    placeholders = ",".join("?" * len(files_hit))
    row = conn.execute(
        f"SELECT COALESCE(SUM(LENGTH(content)), 0) FROM chunks WHERE file IN ({placeholders})",
        files_hit
    ).fetchone()
    chars_full = row[0]
    conn.execute(
        "INSERT INTO search_stats "
        "(ts, source, query, top_k, chunks_ret, chars_ret, files_hit, chars_full, chars_saved) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (datetime.now().isoformat(), source, query[:300], top_k,
         len(results), chars_ret, len(files_hit), chars_full, chars_full - chars_ret)
    )
    conn.commit()


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
        if not body:
            return
        header = f"[{path.stem}] {file_title}"
        if section:
            header += f" > {section}"
        section_key = section or file_title or path.stem

        if len(body) <= CHUNK_MAX_CHARS:
            chunks.append({
                "file": path.name,
                "section": section_key,
                "content": f"{header}\n\n{body}",
            })
            return

        # sezione lunga: spezza in sub-chunk sovrapposti
        step = CHUNK_MAX_CHARS - CHUNK_OVERLAP
        idx = 0
        part = 1
        while idx < len(body):
            slice_text = body[idx: idx + CHUNK_MAX_CHARS]
            chunks.append({
                "file": path.name,
                "section": f"{section_key} [{part}]",
                "content": f"{header} [{part}]\n\n{slice_text}",
            })
            idx += step
            part += 1

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
    embeddings = np.array(list(model.embed(texts)))

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

    # invalida il cache in-memory: verrà ricaricato alla prossima ricerca
    global _emb_matrix, _emb_meta, _bm25
    _emb_matrix = None
    _emb_meta = None
    _bm25 = None

    return updated_files, total_chunks


# ── SEARCH ────────────────────────────────────────────────────────────────────

def rank(query: str, top_k: int = 5, mode: str = None, rerank: bool = True) -> list[dict]:
    """Ranking puro, senza side effect (usato da search() e da eval/).
    Con il reranker attivo ogni risultato porta anche `rerank` (punteggio del cross-encoder)
    e `confidence`: probabilita' calibrata che la risposta sia fra questi risultati."""
    global _emb_matrix, _emb_meta

    # carica cache se non ancora inizializzata
    if _emb_matrix is None:
        conn = get_db()
        _load_emb_cache(conn)
        conn.close()

    if _emb_matrix.shape[0] == 0:
        return []

    model = get_model()
    q_emb = np.array(list(model.embed([query])))[0].astype(np.float32)

    # cosine similarity vectorizzata: una sola matrix multiplication
    norms = np.linalg.norm(_emb_matrix, axis=1)
    q_norm = float(np.linalg.norm(q_emb))
    sem = (_emb_matrix @ q_emb) / (norms * q_norm + 1e-9)

    if (mode or SEARCH_MODE) == "semantic":
        top_idx = np.argsort(sem)[::-1][:top_k]
        return [{**_emb_meta[i], "score": float(sem[i])} for i in top_idx]

    # hybrid: Reciprocal Rank Fusion dei due ranking (le scale dei punteggi non sono confrontabili)
    lex = _bm25.scores(query)
    fused = Counter()
    for ranking in (np.argsort(sem)[::-1][:FUSION_DEPTH], np.argsort(lex)[::-1][:FUSION_DEPTH]):
        for pos, i in enumerate(ranking):
            fused[int(i)] += 1 / (RRF_K + pos + 1)
    reranker = get_reranker() if rerank else False
    candidates = fused.most_common(max(top_k, RERANK_CANDIDATES) if reranker else top_k)
    results = [
        {**_emb_meta[i], "score": score, "sem": float(sem[i]), "bm25": float(lex[i])}
        for i, score in candidates
    ]
    if not reranker:
        return results

    scores = reranker.scores(query, [r["content"] for r in results])
    for r, s in zip(results, scores):
        r["rerank"] = float(s)
    results.sort(key=lambda r: -r["rerank"])
    confidence = reranker.probability(results[0]["rerank"])
    if confidence is not None:
        for r in results:
            r["confidence"] = confidence
    return results[:top_k]


def search(query: str, top_k: int = 5, source: str = "mcp") -> list[dict]:
    results = rank(query, top_k)

    try:
        conn = get_db()
        _record_stat(conn, source, query, top_k, results)
        conn.close()
    except Exception:
        pass

    return results


def format_results(results: list[dict]) -> str:
    if not results:
        return "Nessun risultato trovato nella KB."
    parts = []
    if "confidence" in results[0]:
        c = results[0]["confidence"]
        if c < MIN_CONFIDENCE:
            parts.append(f"⚠️ Confidenza bassa ({c:.2f}): probabilmente la risposta NON è nella KB. "
                         "Non basarti su questi chunk senza verificarli.")
        else:
            parts.append(f"Confidenza: {c:.2f}")
    for i, r in enumerate(results, 1):
        if "rerank" in r:
            score = f"rerank: {r['rerank']:.2f}, rrf: {r['score']:.4f}, bm25: {r['bm25']:.1f}"
        elif "bm25" in r:
            score = f"rrf: {r['score']:.4f}, sem: {r['sem']:.3f}, bm25: {r['bm25']:.1f}"
        else:
            score = f"score: {r['score']:.3f}"
        parts.append(
            f"--- [{r['file']} > {r['section']}] ({score}) ---\n{r['content']}"
        )
    return "\n\n".join(parts)


# ── MCP SERVER (FastMCP) ──────────────────────────────────────────────────────

_server_name = os.environ.get("KB_RAG_NAME", "kb-rag")
mcp = MCPServer(_server_name)


@mcp.tool()
def kb_search(query: str, top_k: int = 5) -> str:
    """
    Search the knowledge base (hybrid: BM25 keyword + semantic similarity).
    Returns the most relevant chunks with source file and section.
    Exact identifiers (table names, ports, hostnames) in the query help ranking.

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


@mcp.tool()
def kb_savings() -> str:
    """
    Mostra le statistiche di risparmio token del RAG.
    Confronta i token effettivamente serviti dai chunk vs la lettura completa dei file sorgente.
    """
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM search_stats").fetchone()[0]
    if not total:
        conn.close()
        return "Nessuna ricerca registrata ancora. Le stats si accumulano durante l'uso."

    agg = conn.execute("""
        SELECT
            COUNT(*)            AS searches,
            SUM(chunks_ret)     AS tot_chunks,
            SUM(chars_ret)      AS tot_chars_ret,
            SUM(chars_full)     AS tot_chars_full,
            SUM(chars_saved)    AS tot_chars_saved,
            AVG(chunks_ret)     AS avg_chunks,
            AVG(files_hit)      AS avg_files,
            MIN(ts)             AS first_ts,
            MAX(ts)             AS last_ts
        FROM search_stats
    """).fetchone()

    by_source = conn.execute("""
        SELECT source, COUNT(*), SUM(chars_ret), SUM(chars_full), SUM(chars_saved)
        FROM search_stats
        GROUP BY source
        ORDER BY COUNT(*) DESC
    """).fetchall()

    conn.close()

    searches, tot_chunks, chars_ret, chars_full, chars_saved, avg_chunks, avg_files, first_ts, last_ts = agg
    chars_ret   = chars_ret   or 0
    chars_full  = chars_full  or 0
    chars_saved = chars_saved or 0

    tok_ret   = chars_ret   // 4
    tok_full  = chars_full  // 4
    tok_saved = chars_saved // 4
    pct = (chars_saved / chars_full * 100) if chars_full else 0

    lines = [
        f"📊 RAG Token Savings — {searches:,} ricerche",
        f"Periodo: {first_ts[:10]} → {last_ts[:10]}",
        "",
        f"  Token serviti dal RAG:        {tok_ret:>10,}",
        f"  Token senza RAG (stima full): {tok_full:>10,}",
        f"  ─────────────────────────────────────────",
        f"  Token risparmiati:            {tok_saved:>10,}  ({pct:.1f}% riduzione)",
        "",
        f"  Media per ricerca: {avg_chunks:.1f} chunk da {avg_files:.1f} file",
        "",
        "Per sorgente:",
    ]
    for source, count, c_ret, c_full, c_saved in by_source:
        t_saved = (c_saved or 0) // 4
        t_full  = (c_full  or 0) // 4
        pct_s   = ((c_saved or 0) / c_full * 100) if c_full else 0
        lines.append(f"  {source:<8} {count:>5} ricerche  →  {t_saved:>8,} tok saved / {t_full:>8,} baseline  ({pct_s:.0f}%)")

    return "\n".join(lines)


# ── STARTUP ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("KB RAG: caricamento modello...", file=sys.stderr)
    get_model()  # scarica modello al primo avvio (~80MB)

    conn = get_db()
    updated, chunks = reindex_all(conn)
    _load_emb_cache(conn)
    conn.close()

    if updated:
        print(f"KB RAG: reindicizzati {updated} file ({chunks} chunk)", file=sys.stderr)
    else:
        print("KB RAG: indice aggiornato, nessun file modificato", file=sys.stderr)

    print(f"KB RAG: cache embedding caricato ({len(_emb_meta)} chunk in RAM)", file=sys.stderr)
    print("KB RAG: server pronto", file=sys.stderr)
    mcp.run()
