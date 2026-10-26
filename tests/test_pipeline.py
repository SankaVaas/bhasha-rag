from bhasharag import RAGPipeline

DOCS = "data/sample"


def test_sinhala_query_retrieves_kandy_passage():
    rag = RAGPipeline(embedder="hash")
    assert rag.index_path(DOCS) > 0
    top, _ = rag.retrieve("දළදා මාළිගාව පිහිටා ඇත්තේ කොහෙද?", k=1)[0]
    assert "දළදා" in top.text


def test_save_load_roundtrip(tmp_path):
    rag = RAGPipeline(embedder="hash")
    rag.index_path(DOCS)
    rag.save(tmp_path)
    again = RAGPipeline.load(tmp_path, embedder="hash")
    assert len(again.store.chunks) == len(rag.store.chunks)
