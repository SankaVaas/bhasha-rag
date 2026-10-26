"""End-to-end pipeline: ingest -> embed -> store -> retrieve -> generate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_text
from .embedder import Embedder, get_embedder
from .generator import get_generator
from .loaders import iter_documents
from .store import VectorStore


@dataclass
class Answer:
    text: str
    sources: list[tuple[Chunk, float]]


class RAGPipeline:
    def __init__(self, embedder: Embedder | str = "e5", generator="extractive", store: VectorStore | None = None):
        self.embedder = get_embedder(embedder) if isinstance(embedder, str) else embedder
        self.generator = get_generator(generator) if isinstance(generator, str) else generator
        self.store = store or VectorStore()

    def index_texts(self, docs: list[tuple[str, str]], max_chars: int = 400) -> int:
        chunks: list[Chunk] = []
        for source, text in docs:
            chunks.extend(chunk_text(text, source=source, max_chars=max_chars))
        if chunks:
            self.store.add(self.embedder.embed_passages([c.text for c in chunks]), chunks)
        return len(chunks)

    def index_path(self, path: str | Path, **kw) -> int:
        return self.index_texts(list(iter_documents(Path(path))), **kw)

    def retrieve(self, question: str, k: int = 4):
        return self.store.search(self.embedder.embed_query(question), k=k)

    def ask(self, question: str, k: int = 4) -> Answer:
        hits = self.retrieve(question, k)
        return Answer(self.generator.generate(question, [c for c, _ in hits]), hits)

    def save(self, directory: str | Path) -> None:
        self.store.save(directory)

    @classmethod
    def load(cls, directory: str | Path, **kw) -> "RAGPipeline":
        return cls(store=VectorStore.load(directory), **kw)
