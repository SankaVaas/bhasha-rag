# bhasha-rag

Offline-first retrieval-augmented generation for **Sinhala** and **Tamil** documents.
Runs on a laptop with no cloud calls: multilingual embeddings for retrieval, an optional
small local LLM (via Ollama) for answers.

## Why
Most RAG tooling assumes English. Sinhala and Tamil have their own scripts, sparse
benchmarks and tokenizers that fragment text badly. This project tackles that end to end
and measures it.

## Quick start
```bash
pip install -e ".[embed,dev]"
bhasha index data/sample --out .index
bhasha ask "දළදා මාළිගාව පිහිටා ඇත්තේ කොහෙද?" --index .index
# with a local LLM:
bhasha ask "புனித தந்தத் தாது கோயில் எங்கே உள்ளது?" --llm ollama:qwen2.5:3b
```
Offline demo with no model download: add `--embedder hash` to both commands.

## Layout
| Path | Purpose |
|---|---|
| `src/bhasharag/lang.py` | script-based language detection |
| `chunking.py` | sentence-aware chunking (handles `.`, `?`, `।`) |
| `embedder.py` | `multilingual-e5` backend and a hash n-gram baseline |
| `store.py` | persistent numpy cosine vector store |
| `generator.py` | Ollama and extractive generators |
| `pipeline.py` / `cli.py` | orchestration and `bhasha` command |
| `eval/` | hit@k and MRR retrieval evaluation |

## Roadmap
- [ ] Build a Sinhala/Tamil QA eval set (50+ questions) and report hit@k per embedder
- [ ] Compare embedders: e5-small, e5-base, BGE-M3, hash baseline
- [ ] Hybrid retrieval: BM25 + dense, with reciprocal rank fusion
- [ ] Sinhala-aware chunking and tokenization analysis (tokens per word across models)
- [ ] Quantized LLM via llama.cpp, with latency and RAM benchmarks
- [ ] Cross-lingual retrieval (Sinhala question, Tamil or English document)
- [ ] Mobile/desktop UI (Streamlit first, then Tauri or Flutter)
- [ ] Answer faithfulness evaluation

## License
MIT

## Evaluation set
`eval/qa.jsonl` holds 50 questions over `data/corpus/` (25 Sinhala, 25 Tamil).

| | fact | list | how_why | number_date | unanswerable |
|---|---|---|---|---|---|
| count | 16 | 12 | 14 | 2 | 6 |

- **Labels:** each answerable question has an `expected_substring` copied verbatim from one sentence of
  its source file. A retrieval hit means a chunk from that file contains it, so the labels stay valid if
  chunk size changes.
- **Splits:** 20 `dev` (tune against this) and 30 `test` (report final numbers only).
- **Unanswerable:** 6 questions whose answer is not in the corpus. They are excluded from hit@k and
  used to study a similarity threshold for refusing to answer.
- **Validate:** `python eval/validate_eval.py` (also runs in `pytest`).
- **Run:** `python eval/retrieval_eval.py --embedder hash --split dev`

### Baseline (hash n-gram embedder, k=4, 400-char chunks)
| split | n | hit@1 | hit@4 | MRR |
|---|---|---|---|---|
| dev | 16 | 0.94 | 0.94 | 0.94 |
| test | 28 | 0.79 | 0.89 | 0.83 |

### Known limitations
- The corpus is small (34 chunks), so hit@4 saturates quickly. Prefer hit@1 and MRR, and grow the corpus.
- Questions were drafted with LLM assistance and need review by native Sinhala and Tamil speakers.
- Every document appears in both languages on the same topic, so cross-language distractors exist.
