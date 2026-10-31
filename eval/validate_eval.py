"""Validate eval/qa.jsonl so a bad label can't silently corrupt results.

Checks: required fields, unique ids/questions, source exists, expected_substring appears
verbatim inside ONE sentence of the source (so it always lands inside a single chunk),
question script matches `lang`, and prints a balance report.

Usage: python eval/validate_eval.py [eval/qa.jsonl]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bhasharag.chunking import split_sentences  # noqa: E402
from bhasharag.lang import detect_language  # noqa: E402

REQUIRED = {"id", "lang", "type", "split", "question", "expected_substring", "source"}
TYPES = {"fact", "list", "how_why", "number_date", "unanswerable"}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def validate(path: Path) -> tuple[list[str], list[dict]]:
    rows, errors = load(path), []
    ids, questions = set(), set()
    for r in rows:
        rid = r.get("id", "?")
        missing = REQUIRED - r.keys()
        if missing:
            errors.append(f"{rid}: missing fields {sorted(missing)}")
            continue
        if rid in ids:
            errors.append(f"{rid}: duplicate id")
        if r["question"] in questions:
            errors.append(f"{rid}: duplicate question")
        ids.add(rid)
        questions.add(r["question"])
        if r["type"] not in TYPES:
            errors.append(f"{rid}: unknown type {r['type']}")
        if r["split"] not in {"dev", "test"}:
            errors.append(f"{rid}: split must be dev/test")
        if detect_language(r["question"]) != r["lang"]:
            errors.append(f"{rid}: question script does not match lang={r['lang']}")
        src = ROOT / r["source"]
        if not src.exists():
            errors.append(f"{rid}: source not found {r['source']}")
            continue
        sub = r["expected_substring"]
        if r["type"] == "unanswerable":
            if sub is not None:
                errors.append(f"{rid}: unanswerable must have expected_substring null")
            continue
        if not sub:
            errors.append(f"{rid}: answerable question needs expected_substring")
            continue
        text = src.read_text(encoding="utf-8")
        if sub not in text:
            errors.append(f"{rid}: substring NOT found verbatim in {src.name}: {sub!r}")
        elif not any(sub in s for s in split_sentences(text)):
            errors.append(f"{rid}: substring spans a sentence boundary (may be split across chunks)")
    return errors, rows


def report(rows: list[dict]) -> None:
    print(f"total: {len(rows)}")
    for key in ("lang", "type", "split"):
        print(f"  by {key}: {dict(sorted(Counter(r[key] for r in rows).items()))}")
    print(f"  by lang x split: {dict(sorted(Counter((r['lang'], r['split']) for r in rows).items()))}")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "qa.jsonl"
    errors, rows = validate(path)
    report(rows)
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print("  -", e)
        return 1
    print("\nOK: all labels verified against the source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
