import pytest

from tiny_transformer.attention import MultiHeadCausalSelfAttention, causal_mask
from tiny_transformer.config import TinyGPTConfig


def test_causal_mask_is_lower_triangular() -> None:
    torch = pytest.importorskip("torch")

    mask = causal_mask(seq_len=4)
    expected = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            ]
        ]
    )

    assert mask.shape == (1, 1, 4, 4)
    assert torch.equal(mask, expected)


def test_multi_head_causal_self_attention_preserves_input_shape() -> None:
    torch = pytest.importorskip("torch")

    config = TinyGPTConfig(n_embd=8, n_head=2)
    attn = MultiHeadCausalSelfAttention(config)
    x = torch.randn(2, 5, config.n_embd)

    out = attn(x)

    assert out.shape == x.shape


def test_multi_head_causal_self_attention_does_not_use_future_tokens() -> None:
    torch = pytest.importorskip("torch")

    config = TinyGPTConfig(n_embd=8, n_head=2)
    attn = MultiHeadCausalSelfAttention(config)

    x1 = torch.randn(1, 4, config.n_embd)
    x2 = x1.clone()
    x2[:, -1, :] = x2[:, -1, :] + 100.0

    out1 = attn(x1)
    out2 = attn(x2)

    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-6)
    assert not torch.allclose(out1[:, -1, :], out2[:, -1, :], atol=1e-6)
