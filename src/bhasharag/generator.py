"""Answer generation: local LLM through Ollama, or an extractive fallback."""
from __future__ import annotations

import json
import urllib.request

from .chunking import Chunk
from .lang import detect_language

LANG_NAME = {"si": "Sinhala", "ta": "Tamil", "en": "English"}


def build_prompt(question: str, contexts: list[Chunk]) -> str:
    lang = LANG_NAME[detect_language(question)]
    ctx = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(contexts))
    return (
        f"Answer the question using ONLY the context below. Reply in {lang}. "
        f"If the context does not contain the answer, say you do not know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


class ExtractiveGenerator:
    """No LLM: return the best-matching passage. Useful as a baseline and offline fallback."""

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        return contexts[0].text if contexts else ""


class OllamaGenerator:
    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
        self.model, self.host = model, host.rstrip("/")

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        payload = json.dumps(
            {"model": self.model, "prompt": build_prompt(question, contexts), "stream": False}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["response"].strip()


def get_generator(spec: str):
    if spec == "extractive":
        return ExtractiveGenerator()
    if spec.startswith("ollama:"):
        return OllamaGenerator(model=spec.split(":", 1)[1])
    raise ValueError("Use 'extractive' or 'ollama:<model>'")
