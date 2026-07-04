import pytest

torch = pytest.importorskip("torch")

from tiny_transformer.config import TinyGPTConfig
from tiny_transformer.model import TinyGPT


def test_tiny_gpt_returns_logits_for_each_position_and_vocab_item() -> None:
    config = TinyGPTConfig(vocab_size=16, block_size=6, n_layer=1, n_head=2, n_embd=8)
    model = TinyGPT(config)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    logits = model(input_ids)

    assert logits.shape == (2, 3, config.vocab_size)


def test_tiny_gpt_rejects_sequence_longer_than_block_size() -> None:
    config = TinyGPTConfig(vocab_size=16, block_size=3, n_layer=1, n_head=2, n_embd=8)
    model = TinyGPT(config)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    with pytest.raises(ValueError, match="sequence length exceeds block_size"):
        model(input_ids)
