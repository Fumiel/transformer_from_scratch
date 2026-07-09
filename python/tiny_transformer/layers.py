"""Basic layers for the Python implementation.

TODO:
- Implement Linear without using nn.Linear if you want maximum learning effect.
- Implement LayerNorm and GELU.
- Compare each layer with PyTorch reference outputs.
"""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # Allow tokenizer tests to run before installing torch.
    torch = None
    nn = object  # type: ignore[assignment]


if torch is not None:

    class Linear(nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
            super().__init__()
            # 入力が in_features 次元、出力が out_features 次元の線形変換を行う層。
            # nn.Parameter は学習可能なパラメータを定義するためのクラス。
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            # バイアスは通常、学習開始時に大きくずれないように0で初期化される。
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
            # ランダム値。
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [..., in_features]
            # t()は転置を意味する。@は行列積を意味する。
            y = x @ self.weight.t()
            if self.bias is not None:
                y = y + self.bias
            return y


    class LayerNorm(nn.Module):
        def __init__(self, n_embd: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(n_embd))
            self.bias = nn.Parameter(torch.zeros(n_embd))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, C]
            # 平均、dim=-1は最後の次元に沿って計算することを意味する。keepdim=Trueは、平均後も次元を保持することを意味する。
            mean = x.mean(dim=-1, keepdim=True)
            # 分散、unbiased=Falseは不偏分散ではなく母分散を計算することを意味する。
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight * x_hat + self.bias


    def gelu(x: torch.Tensor) -> torch.Tensor:
        # Approximate GELU used by GPT-style models.
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
