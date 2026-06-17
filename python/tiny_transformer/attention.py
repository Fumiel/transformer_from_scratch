"""Attention modules.

The main exercise is to implement MultiHeadCausalSelfAttention and verify shapes.
"""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    from .config import TinyGPTConfig
    from .layers import Linear
except ModuleNotFoundError:
    torch = None
    nn = object  # type: ignore[assignment]


if torch is not None:

    def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
        # shape: [1, 1, T, T]
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)


    def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # q, k, v: [B, H, T, D]
        scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
        scores = scores / math.sqrt(q.size(-1))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return probs @ v  # [B, H, T, D]


    class MultiHeadCausalSelfAttention(nn.Module):
        def __init__(self, config: TinyGPTConfig) -> None:
            super().__init__()
            assert config.n_embd % config.n_head == 0
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.head_dim = config.n_embd // config.n_head

            self.qkv_proj = Linear(config.n_embd, 3 * config.n_embd)
            self.out_proj = Linear(config.n_embd, config.n_embd)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, T, C]
            bsz, seq_len, channels = x.shape
            qkv = self.qkv_proj(x)  # [B, T, 3C]
            q, k, v = qkv.split(channels, dim=-1)

            # [B, T, C] -> [B, H, T, D]
            q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

            mask = causal_mask(seq_len, device=x.device)
            out = scaled_dot_product_attention(q, k, v, mask)  # [B, H, T, D]

            # [B, H, T, D] -> [B, T, C]
            out = out.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
            return self.out_proj(out)
