from __future__ import annotations

import argparse

from .pipeline import RAGPipeline


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="bhasha", description="Sinhala/Tamil offline RAG")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="index a file or folder")
    pi.add_argument("path")
    pi.add_argument("--out", default=".index")
    pi.add_argument("--embedder", default="e5", choices=["e5", "hash"])

    pa = sub.add_parser("ask", help="ask a question")
    pa.add_argument("question")
    pa.add_argument("--index", default=".index")
    pa.add_argument("--embedder", default="e5", choices=["e5", "hash"])
    pa.add_argument("--llm", default="extractive", help="'extractive' or 'ollama:<model>'")
    pa.add_argument("-k", type=int, default=4)

    args = p.parse_args(argv)
    if args.cmd == "index":
        rag = RAGPipeline(embedder=args.embedder)
        n = rag.index_path(args.path)
        rag.save(args.out)
        print(f"Indexed {n} chunks -> {args.out}")
    else:
        rag = RAGPipeline.load(args.index, embedder=args.embedder, generator=args.llm)
        ans = rag.ask(args.question, k=args.k)
        print(ans.text)
        print("\nSources:")
        for chunk, score in ans.sources:
            print(f"  [{score:.3f}] {chunk.source}#{chunk.index} ({chunk.lang})")


if __name__ == "__main__":
    main()
