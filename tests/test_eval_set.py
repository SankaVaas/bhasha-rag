import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from validate_eval import validate  # noqa: E402


def test_eval_set_labels_are_valid():
    errors, rows = validate(ROOT / "eval" / "qa.jsonl")
    assert errors == []
    assert len(rows) >= 50
