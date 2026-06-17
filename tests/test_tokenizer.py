from tiny_transformer import CharTokenizer


def test_char_tokenizer_roundtrip() -> None:
    text = "hello transformer"
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_char_tokenizer_vocab_size() -> None:
    tokenizer = CharTokenizer("abca")
    assert tokenizer.vocab_size == 3
