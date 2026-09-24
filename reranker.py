# Copyright (c) 2026 Sergio Angelastro — MIT License
"""
Secondo stadio del retrieval: un cross-encoder locale riordina i candidati della ricerca ibrida
e ne stima la probabilita' di rilevanza.

E' il ruolo che avrebbe un "System One model" (es. TypeSafe Jev): una sola passata sul modello,
nessun testo generato, un punteggio per ogni coppia (query, chunk) calcolato in parallelo.
Qui gira in locale su ONNX Runtime: i chunk della KB non escono dalla macchina.

La probabilita' e' calibrata con Platt scaling (p = sigmoid(a·score + b)): i parametri per modello
si ricavano con `py eval/eval_retrieval.py --calibrate` sul proprio gold set.
"""
import math
import os
from pathlib import Path

import numpy as np

# Solo modelli multilingua con licenza che ammette l'uso commerciale.
MODELS = {
    # default: sulla KB di riferimento e' il piu' preciso al 1° posto ed e' 5x piu' veloce di bge-m3
    "mmarco-mminilm": {
        "hf": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "file": "onnx/model_qint8_avx512_vnni.onnx",   # int8, ~45% piu' veloce su CPU con AVX-512 VNNI
        "license": "apache-2.0",
        "size_gb": 0.12,
        "platt": (0.392, 0.356),   # 5-fold CV su gold set di 57 query: acc 0.95, ECE 0.07
    },
    "mmarco-mminilm-avx2": {
        "hf": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "file": "onnx/model_quint8_avx2.onnx",         # stesso modello, per CPU senza AVX-512
        "license": "apache-2.0",
        "size_gb": 0.12,
        "platt": (0.392, 0.356),   # 5-fold CV su gold set di 57 query: acc 0.95, ECE 0.07
    },
    "bge-m3": {
        "hf": "onnx-community/bge-reranker-v2-m3-ONNX",
        "file": "onnx/model_int8.onnx",
        "license": "apache-2.0",
        "size_gb": 0.57,
        "platt": None,             # non calibrato: niente confidenza, solo riordino
    },
}

DEFAULT_CACHE = Path(os.environ.get("KB_RERANK_CACHE", str(Path.home() / ".cache" / "claude-rag" / "models")))


class Reranker:
    def __init__(self, name: str, threads: int = 4, cache_dir: Path = DEFAULT_CACHE):
        from fastembed.common.model_description import ModelSource
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        spec = MODELS[name]
        # fastembed registra i modelli per nome: stesso repo HF con file diversi = alias diversi
        alias = f"claude-rag/{name}"
        known = {m["model"] for m in TextCrossEncoder.list_supported_models()}
        if alias not in known:
            TextCrossEncoder.add_custom_model(
                model=alias, sources=ModelSource(hf=spec["hf"]), model_file=spec["file"],
                license=spec["license"], size_in_gb=spec["size_gb"],
            )
        self.name = name
        self.platt = spec["platt"]
        self.model = TextCrossEncoder(model_name=alias, cache_dir=str(cache_dir), threads=threads)

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        return np.array(list(self.model.rerank(query, texts)), dtype=np.float32)

    def probability(self, score: float) -> float | None:
        """Probabilita' che la risposta sia fra i primi risultati, dato il punteggio del migliore."""
        if self.platt is None:
            return None
        a, b = self.platt
        z = max(-30.0, min(30.0, a * score + b))
        return 1 / (1 + math.exp(-z))
