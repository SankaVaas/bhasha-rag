"""Tiny persistent vector store: cosine search over a numpy matrix."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, vectors: np.ndarray | None = None, chunks: list[Chunk] | None = None):
        self.vectors = vectors if vectors is not None else np.zeros((0, 0), dtype=np.float32)
        self.chunks = chunks or []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        self.vectors = vectors if len(self.chunks) == 0 else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        scores = self.vectors @ query_vec
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        d = Path(directory)
        vectors = np.load(d / "vectors.npy")
        chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
        return cls(vectors, chunks)
