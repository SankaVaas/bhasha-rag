"""Script-based language detection for Sinhala, Tamil and English."""
from __future__ import annotations

SINHALA = (0x0D80, 0x0DFF)
TAMIL = (0x0B80, 0x0BFF)


def detect_language(text: str) -> str:
    """Return 'si', 'ta' or 'en' by counting characters in each Unicode block."""
    si = ta = latin = 0
    for ch in text:
        cp = ord(ch)
        if SINHALA[0] <= cp <= SINHALA[1]:
            si += 1
        elif TAMIL[0] <= cp <= TAMIL[1]:
            ta += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    best = max((si, "si"), (ta, "ta"), (latin, "en"))
    return best[1] if best[0] > 0 else "en"
