"""
Baseline di retrieval della KB contro il gold set.

Valuta, sugli stessi chunk indicizzati in kb.db:
  - semantic : quello che fa oggi kb_search (MiniLM + coseno)
  - lexical  : quello che fa oggi l'hook kb_recall (overlap token / sqrt(len))
  - bm25     : BM25 classico sui chunk (alternativa locale, nessuna dipendenza)
  - hybrid   : fusione RRF semantic + bm25
  - server   : server.rank() — il codice vero di kb_search (deve coincidere con hybrid)
  - rerank   : top 20 di server.rank() riordinati da un cross-encoder locale multilingua
               (proxy di un reranker tipo Jev: stesso compito, dati che non escono)

Non scrive in search_stats: la ricerca e' replicata qui, non chiama server.search().

Uso:  py eval/eval_retrieval.py [--no-rerank]     (dalla root del repo)
      KB_RAG_DB=<path> per usare un kb.db diverso da quello di produzione
"""
import json
import os
import math
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAG = HERE.parent
# stessi default di server.py; kb.db viene solo letto
KB_DB = Path(os.environ.setdefault("KB_RAG_DB", str(RAG / "kb.db")))
os.environ.setdefault("KB_RAG_DIR", str(RAG.parent))
# gold set e risultati contengono contenuti della KB: restano locali (gitignored)
GOLD = HERE / "gold_set.json"
sys.path.insert(0, str(RAG / "hooks"))
sys.path.insert(0, str(RAG))
from kb_recall import tokenize, score_chunk  # noqa: E402

KS = (1, 3, 5, 10)
DEPTH = 50


def load_chunks():
    conn = sqlite3.connect(KB_DB)
    rows = conn.execute("SELECT file, section, content, embedding FROM chunks").fetchall()
    conn.close()
    meta = [{"file": r[0], "section": r[1], "content": r[2]} for r in rows]
    emb = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    return meta, emb


def is_hit(chunk, gold):
    for g in gold:
        if chunk["file"] != g["file"]:
            continue
        if g["section"] is None or chunk["section"].startswith(g["section"]):
            return True
    return False


def is_file_hit(chunk, gold):
    return any(chunk["file"] == g["file"] for g in gold)


# ── retriever ────────────────────────────────────────────────────────────────

class Semantic:
    name = "semantic"

    def __init__(self, meta, emb):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    def rank(self, q):
        v = np.array(list(self.model.embed([q])))[0].astype(np.float32)
        s = self.emb @ (v / (np.linalg.norm(v) + 1e-9))
        idx = np.argsort(s)[::-1][:DEPTH]
        return [(int(i), float(s[i])) for i in idx]


class Lexical:
    name = "lexical"

    def __init__(self, meta):
        self.contents = [m["content"] for m in meta]

    def rank(self, q):
        qt = tokenize(q)
        s = np.array([score_chunk(qt, c) for c in self.contents])
        idx = np.argsort(s)[::-1][:DEPTH]
        return [(int(i), float(s[i])) for i in idx]


class BM25:
    name = "bm25"

    def __init__(self, meta, k1=1.2, b=0.75):
        self.docs = [re.findall(r"\w{2,}", m["content"].lower()) for m in meta]
        self.tf = [Counter(d) for d in self.docs]
        self.len = np.array([len(d) for d in self.docs])
        self.avg = self.len.mean()
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def rank(self, q):
        qt = set(re.findall(r"\w{2,}", q.lower()))
        s = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for t in qt:
                f = tf.get(t)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        idx = np.argsort(s)[::-1][:DEPTH]
        return [(int(i), float(s[i])) for i in idx]


class Hybrid:
    name = "hybrid"

    def __init__(self, a, b, k=60):
        self.a, self.b, self.k = a, b, k

    def rank(self, q):
        fused = Counter()
        for r in (self.a.rank(q), self.b.rank(q)):
            for pos, (i, _) in enumerate(r):
                fused[i] += 1 / (self.k + pos + 1)
        return fused.most_common(DEPTH)


class Server:
    """server.rank() indicizza gli stessi chunk nello stesso ordine di load_chunks()."""
    name = "server"

    def __init__(self, meta):
        import server
        self.server = server
        self.key = {(m["file"], m["section"], m["content"]): i for i, m in enumerate(meta)}

    def rank(self, q, depth=DEPTH):
        return [(self.key[(r["file"], r["section"], r["content"])], r["score"])
                for r in self.server.rank(q, depth, mode="hybrid")]


class Rerank:
    name = "rerank"
    MODEL = "jinaai/jina-reranker-v2-base-multilingual"
    CANDIDATES = 20

    def __init__(self, first_stage, meta):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model = TextCrossEncoder(model_name=self.MODEL)
        self.first, self.meta = first_stage, meta

    def rank(self, q):
        cand = [i for i, _ in self.first.rank(q)[:self.CANDIDATES]]
        scores = list(self.model.rerank(q, [self.meta[i]["content"] for i in cand]))
        return sorted(zip(cand, map(float, scores)), key=lambda x: -x[1])


# ── metriche ─────────────────────────────────────────────────────────────────

def evaluate(retriever, items, meta):
    per_item = []
    for it in items:
        ranked = retriever.rank(it["q"])
        first_sec = next((p for p, (i, _) in enumerate(ranked) if is_hit(meta[i], it["gold"])), None)
        first_file = next((p for p, (i, _) in enumerate(ranked) if is_file_hit(meta[i], it["gold"])), None)
        per_item.append({
            "id": it["id"], "kind": it["kind"], "q": it["q"],
            "first_sec": first_sec, "first_file": first_file,
            "top1_score": ranked[0][1],
            "top5": [f'{meta[i]["file"]} > {meta[i]["section"]} ({s:.3f})' for i, s in ranked[:5]],
        })
    return per_item


def summarize(per_item):
    pos = [p for p in per_item if p["kind"] != "neg"]
    out = {"n": len(pos)}
    for k in KS:
        out[f"sec@{k}"] = sum(p["first_sec"] is not None and p["first_sec"] < k for p in pos) / len(pos)
        out[f"file@{k}"] = sum(p["first_file"] is not None and p["first_file"] < k for p in pos) / len(pos)
    out["mrr"] = sum(1 / (p["first_sec"] + 1) for p in pos if p["first_sec"] is not None) / len(pos)
    for kind in ("log", "nl"):
        sub = [p for p in pos if p["kind"] == kind]
        out[f"sec@5_{kind}"] = sum(p["first_sec"] is not None and p["first_sec"] < 5 for p in sub) / len(sub)
    return out


def abstention(per_item):
    """Una soglia sul top-1 score separa 'risposta c'e' ed e' in top 5' da 'non c'e''?"""
    good = [p["top1_score"] for p in per_item if p["kind"] != "neg" and p["first_sec"] is not None and p["first_sec"] < 5]
    bad = [p["top1_score"] for p in per_item if p["kind"] == "neg" or p["first_sec"] is None or p["first_sec"] >= 5]
    best = None
    for t in sorted(set(good + bad)):
        acc = (sum(g >= t for g in good) + sum(b < t for b in bad)) / (len(good) + len(bad))
        if best is None or acc > best[1]:
            best = (t, acc)
    negs = [p["top1_score"] for p in per_item if p["kind"] == "neg"]
    return {
        "good_min": min(good), "good_median": float(np.median(good)),
        "bad_max": max(bad), "bad_median": float(np.median(bad)),
        "neg_scores": negs,
        "best_threshold": best[0], "best_accuracy": best[1],
        "overlap": sum(g <= max(bad) for g in good) / len(good),
    }


def main():
    if not GOLD.exists():
        sys.exit(f"Manca {GOLD.name}: copia gold_set.example.json e scrivi le query sulla tua KB.")
    items = json.loads(GOLD.read_text(encoding="utf-8"))["items"]
    meta, emb = load_chunks()

    # sanity: ogni gold deve esistere nell'indice
    missing = [(it["id"], g) for it in items for g in it["gold"]
               if not any(is_hit(m, [g]) for m in meta)]
    if missing:
        print("GOLD NON TROVATI NELL'INDICE:", *missing, sep="\n  ")
        sys.exit(1)

    sem, bm, srv = Semantic(meta, emb), BM25(meta), Server(meta)
    retrievers = [sem, Lexical(meta), bm, Hybrid(sem, bm), srv]
    if "--no-rerank" not in sys.argv:
        retrievers.append(Rerank(srv, meta))

    report = {"n_chunks": len(meta), "n_items": len(items), "results": {}}
    for r in retrievers:
        per_item = evaluate(r, items, meta)
        report["results"][r.name] = {
            "summary": summarize(per_item),
            "abstention": abstention(per_item),
            "items": per_item,
        }

    (HERE / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    cols = ["sec@1", "sec@3", "sec@5", "sec@10", "file@1", "file@5", "mrr", "sec@5_log", "sec@5_nl"]
    print(f"{'retriever':<10}" + "".join(f"{c:>10}" for c in cols))
    for name, r in report["results"].items():
        print(f"{name:<10}" + "".join(f"{r['summary'][c]:>10.2f}" for c in cols))
    print()
    for name, r in report["results"].items():
        a = r["abstention"]
        print(f"{name:<10} good(min/med)={a['good_min']:.3f}/{a['good_median']:.3f}  "
              f"bad(med/max)={a['bad_median']:.3f}/{a['bad_max']:.3f}  "
              f"soglia={a['best_threshold']:.3f} acc={a['best_accuracy']:.2f}  "
              f"neg={[round(x, 3) for x in a['neg_scores']]}")


if __name__ == "__main__":
    main()
