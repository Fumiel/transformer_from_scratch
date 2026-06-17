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
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [..., in_features]
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
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight * x_hat + self.bias


    def gelu(x: torch.Tensor) -> torch.Tensor:
        # Approximate GELU used by GPT-style models.
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
