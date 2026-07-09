import pytest

from tiny_transformer.layers import LayerNorm, Linear


def test_linear_forward_applies_weight_and_bias() -> None:
    torch = pytest.importorskip("torch")

    layer = Linear(in_features=3, out_features=2, bias=True)
    x = torch.tensor([[1.0, 2.0, 3.0]])

    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0, 2.0], [-1.0, 2.0, 1.0]]))
        layer.bias.copy_(torch.tensor([0.5, -0.25]))

    out = layer(x)

    expected = torch.tensor([[7.5, 5.75]])
    assert torch.allclose(out, expected)


def test_layernorm_normalizes_along_last_dimension() -> None:
    torch = pytest.importorskip("torch")

    layer = LayerNorm(n_embd=2)
    x = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])

    out = layer(x)

    assert torch.allclose(out.mean(dim=-1), torch.zeros_like(out.mean(dim=-1)), atol=1e-6)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones_like(out.var(dim=-1, unbiased=False)), atol=1e-6)
