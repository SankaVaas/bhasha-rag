"""Embedding backends.

E5Embedder  : multilingual-e5 via sentence-transformers (covers Sinhala + Tamil).
HashEmbedder: dependency-free character n-gram hashing, for tests and offline demos.
"""
from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-9)).astype(np.float32)


class E5Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts):
        return _normalize(self.model.encode([f"passage: {t}" for t in texts]))

    def embed_query(self, text):
        return _normalize(self.model.encode([f"query: {text}"]))[0]


class HashEmbedder:
    def __init__(self, dim: int = 1024, ngram: tuple[int, ...] = (2, 3, 4)):
        self.dim, self.ngram = dim, ngram

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = f" {text.lower()} "
        for n in self.ngram:
            for i in range(len(t) - n + 1):
                h = int(hashlib.md5(t[i : i + n].encode("utf-8")).hexdigest(), 16)
                v[h % self.dim] += 1.0
        return v

    def embed_passages(self, texts):
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text):
        return _normalize(self._vec(text)[None, :])[0]


def get_embedder(name: str) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "e5":
        return E5Embedder()
    raise ValueError(f"Unknown embedder '{name}' (use 'e5' or 'hash')")
