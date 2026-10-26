"""Sentence-aware chunking that works for Sinhala, Tamil and English."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .lang import detect_language

_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+|\n{2,}")


@dataclass
class Chunk:
    text: str
    source: str
    index: int
    lang: str


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def chunk_text(text: str, source: str = "", max_chars: int = 400, overlap_sentences: int = 1) -> list[Chunk]:
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    size = 0

    def flush():
        if current:
            body = " ".join(current)
            chunks.append(Chunk(body, source, len(chunks), detect_language(body)))

    for sent in sentences:
        if current and size + len(sent) > max_chars:
            flush()
            current = current[-overlap_sentences:] if overlap_sentences else []
            size = sum(len(s) for s in current)
        current.append(sent)
        size += len(sent)
    flush()
    return chunks
