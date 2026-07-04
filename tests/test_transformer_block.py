import pytest

torch = pytest.importorskip("torch")

from tiny_transformer.config import TinyGPTConfig
from tiny_transformer.model import TransformerBlock


def test_transformer_block_preserves_shape() -> None:
    config = TinyGPTConfig(n_embd=8, n_head=2)
    block = TransformerBlock(config)
    x = torch.randn(2, 5, config.n_embd)

    out = block(x)

    assert out.shape == x.shape


def test_transformer_block_uses_residual_path_when_sublayers_are_zeroed() -> None:
    config = TinyGPTConfig(n_embd=8, n_head=2)
    block = TransformerBlock(config)
    x = torch.randn(2, 5, config.n_embd)

    # 勾配計算をせずに全てのパラメータを0にする。
    with torch.no_grad():
        for param in block.parameters():
            param.zero_()

    out = block(x)

    assert torch.allclose(out, x) # ほぼ等しいか？
