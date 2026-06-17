from __future__ import annotations


class CharTokenizer:
    """Simple character-level tokenizer.

    This tokenizer is intentionally small and transparent.
    It is useful for the first implementation round because every token is visible.
    """

    def __init__(self, text: str | None = None) -> None:
        if text is None:
            chars = [chr(i) for i in range(128)]
        else:
            chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"unknown character: {exc.args[0]!r}") from exc

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.itos[i] for i in ids)
        except KeyError as exc:
            raise ValueError(f"unknown token id: {exc.args[0]!r}") from exc
