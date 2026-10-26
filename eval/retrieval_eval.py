"""Retrieval quality: hit@k and MRR over a JSONL of questions.

Usage: python eval/retrieval_eval.py eval/qa_sample.jsonl --data data/sample --embedder hash
"""
import argparse
import json

from bhasharag import RAGPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qa")
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--embedder", default="e5")
    ap.add_argument("-k", type=int, default=4)
    a = ap.parse_args()

    rag = RAGPipeline(embedder=a.embedder)
    rag.index_path(a.data)
    rows = [json.loads(l) for l in open(a.qa, encoding="utf-8") if l.strip()]
    hits, rr = 0, 0.0
    for r in rows:
        results = rag.retrieve(r["question"], a.k)
        for rank, (chunk, _) in enumerate(results, 1):
            if r["expected_substring"] in chunk.text:
                hits += 1
                rr += 1 / rank
                break
    n = len(rows)
    print(f"hit@{a.k}: {hits / n:.2f}   MRR: {rr / n:.2f}   (n={n})")


if __name__ == "__main__":
    main()
