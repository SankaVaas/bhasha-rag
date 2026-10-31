"""Retrieval evaluation over eval/qa.jsonl: hit@1, hit@k and MRR, broken down by
language, question type and split. Unanswerable questions are excluded from hit/MRR and
instead reported as top-1 similarity scores vs answerable ones (a basis for a refusal threshold).

A hit = a retrieved chunk from the right source file contains expected_substring.

Usage:
  python eval/retrieval_eval.py --embedder hash --split dev
  python eval/retrieval_eval.py --embedder e5 --split dev -k 4 --max-chars 400
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from bhasharag import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent


def first_hit_rank(results, row):
    want = Path(row["source"]).name
    for rank, (chunk, _) in enumerate(results, 1):
        if Path(chunk.source).name == want and row["expected_substring"] in chunk.text:
            return rank
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=str(ROOT / "eval" / "qa.jsonl"))
    ap.add_argument("--data", default=str(ROOT / "data" / "corpus"))
    ap.add_argument("--embedder", default="e5", choices=["e5", "hash"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=400)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.qa, encoding="utf-8") if l.strip()]
    if a.split != "all":
        rows = [r for r in rows if r["split"] == a.split]

    rag = RAGPipeline(embedder=a.embedder)
    n_chunks = rag.index_path(a.data, max_chars=a.max_chars)

    groups = defaultdict(lambda: {"n": 0, "h1": 0, "hk": 0, "rr": 0.0})
    top = {"answerable": [], "unanswerable": []}
    for r in rows:
        results = rag.retrieve(r["question"], a.k)
        if r["type"] == "unanswerable":
            top["unanswerable"].append(results[0][1])
            continue
        top["answerable"].append(results[0][1])
        rank = first_hit_rank(results, r)
        for key in ("ALL", f"lang={r['lang']}", f"type={r['type']}"):
            g = groups[key]
            g["n"] += 1
            if rank:
                g["hk"] += 1
                g["rr"] += 1 / rank
                g["h1"] += rank == 1

    print(f"embedder={a.embedder} split={a.split} k={a.k} max_chars={a.max_chars} chunks={n_chunks}")
    print(f"{'group':<18}{'n':>4}{'hit@1':>8}{'hit@'+str(a.k):>8}{'MRR':>8}")
    for key in sorted(groups, key=lambda s: (s != "ALL", s)):
        g = groups[key]
        print(f"{key:<18}{g['n']:>4}{g['h1']/g['n']:>8.2f}{g['hk']/g['n']:>8.2f}{g['rr']/g['n']:>8.2f}")
    for name, vals in top.items():
        if vals:
            print(f"top-1 score, {name:<13} n={len(vals):<3} mean={statistics.mean(vals):.3f} "
                  f"min={min(vals):.3f} max={max(vals):.3f}")


if __name__ == "__main__":
    main()
