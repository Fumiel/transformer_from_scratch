import pytest

torch = pytest.importorskip("torch")

from tiny_transformer.config import TinyGPTConfig
from tiny_transformer.model import FeedForward


def test_feedforward_preserves_batch_and_sequence_shape() -> None:
    config = TinyGPTConfig(n_embd=8)
    ffn = FeedForward(config)
    x = torch.randn(2, 5, config.n_embd)

    out = ffn(x)

    assert out.shape == x.shape


def test_feedforward_expands_hidden_dimension_then_projects_back() -> None:
    config = TinyGPTConfig(n_embd=8)
    ffn = FeedForward(config)

    assert ffn.fc1.weight.shape == (4 * config.n_embd, config.n_embd)
    assert ffn.fc2.weight.shape == (config.n_embd, 4 * config.n_embd)
