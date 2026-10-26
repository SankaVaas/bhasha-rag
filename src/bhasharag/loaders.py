"""Load plain text, markdown and PDF files."""
from __future__ import annotations

from pathlib import Path

TEXT_EXT = {".txt", ".md"}


def load_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXT:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path}")


def iter_documents(root: Path):
    """Yield (source_name, text) for every supported file under root."""
    root = Path(root)
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    for f in files:
        if f.is_file() and f.suffix.lower() in TEXT_EXT | {".pdf"}:
            text = load_file(f).strip()
            if text:
                yield str(f), text
