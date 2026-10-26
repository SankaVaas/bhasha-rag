from bhasharag.chunking import chunk_text, split_sentences


def test_splits_on_sentence_marks():
    assert len(split_sentences("එක. දෙක. තුන.")) == 3


def test_chunks_respect_max_chars_and_lang():
    text = " ".join(["கண்டி ஒரு நகரம்."] * 30)
    chunks = chunk_text(text, source="t", max_chars=100)
    assert len(chunks) > 1
    assert all(c.lang == "ta" for c in chunks)
