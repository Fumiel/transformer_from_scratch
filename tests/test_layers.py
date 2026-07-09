import pytest

from tiny_transformer.layers import LayerNorm, Linear, gelu


def test_linear_forward_matches_torch_reference_for_batched_input() -> None:
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")

    layer = Linear(in_features=3, out_features=2, bias=True)
    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]],
            [[2.5, 0.5, -1.0], [3.0, 3.0, 3.0]],
        ]
    )

    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 0.0, 2.0], [-1.0, 2.0, 1.0]]))
        layer.bias.copy_(torch.tensor([0.5, -0.25]))

    out = layer(x)
    expected = functional.linear(x, layer.weight, layer.bias)

    assert torch.allclose(out, expected)


def test_linear_forward_matches_torch_reference_without_bias() -> None:
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")

    layer = Linear(in_features=2, out_features=3, bias=False)
    x = torch.tensor([[1.0, -2.0], [0.5, 4.0]])

    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 3.0], [-2.0, 0.5], [0.0, -1.0]]))

    out = layer(x)
    expected = functional.linear(x, layer.weight, None)

    assert layer.bias is None
    assert torch.allclose(out, expected)


def test_layernorm_matches_torch_reference_with_affine_parameters() -> None:
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")

    layer = LayerNorm(n_embd=3, eps=1e-5)
    x = torch.tensor(
        [
            [[1.0, 3.0, 5.0], [2.0, 4.0, 8.0]],
            [[-1.0, 0.0, 1.0], [10.0, 10.5, 11.0]],
        ]
    )

    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.5, 0.5, -2.0]))
        layer.bias.copy_(torch.tensor([0.25, -1.0, 2.0]))

    out = layer(x)
    expected = functional.layer_norm(x, (3,), layer.weight, layer.bias, layer.eps)

    assert torch.allclose(out, expected, atol=1e-6)


def test_layernorm_constant_input_returns_bias_only() -> None:
    torch = pytest.importorskip("torch")

    layer = LayerNorm(n_embd=4)
    x = torch.full((2, 3, 4), 7.0)

    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        layer.bias.copy_(torch.tensor([-1.0, 0.5, 2.0, 3.5]))

    out = layer(x)
    expected = layer.bias.view(1, 1, 4).expand_as(out)

    assert torch.allclose(out, expected, atol=1e-6)


def test_gelu_matches_torch_tanh_approximation() -> None:
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")

    x = torch.linspace(-3.0, 3.0, steps=13)

    out = gelu(x)
    expected = functional.gelu(x, approximate="tanh")

    assert torch.allclose(out, expected, atol=1e-6)
