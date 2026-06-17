"""Tiny decoder-only Transformer.

This file is intentionally a scaffold.
Start by implementing TransformerBlock and TinyGPT forward pass.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from .attention import MultiHeadCausalSelfAttention
    from .config import TinyGPTConfig
    from .layers import LayerNorm, Linear, gelu
except ModuleNotFoundError:
    torch = None
    nn = object  # type: ignore[assignment]


if torch is not None:

    class FeedForward(nn.Module):
        def __init__(self, config: TinyGPTConfig) -> None:
            super().__init__()
            self.fc1 = Linear(config.n_embd, 4 * config.n_embd)
            self.fc2 = Linear(4 * config.n_embd, config.n_embd)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(gelu(self.fc1(x)))


    class TransformerBlock(nn.Module):
        def __init__(self, config: TinyGPTConfig) -> None:
            super().__init__()
            self.ln1 = LayerNorm(config.n_embd)
            self.attn = MultiHeadCausalSelfAttention(config)
            self.ln2 = LayerNorm(config.n_embd)
            self.ffn = FeedForward(config)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x


    class TinyGPT(nn.Module):
        def __init__(self, config: TinyGPTConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
            self.final_ln = LayerNorm(config.n_embd)
            self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # input_ids: [B, T]
            bsz, seq_len = input_ids.shape
            if seq_len > self.config.block_size:
                raise ValueError("sequence length exceeds block_size")

            positions = torch.arange(seq_len, device=input_ids.device)  # [T]
            x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
            for block in self.blocks:
                x = block(x)
            x = self.final_ln(x)
            logits = self.lm_head(x)  # [B, T, vocab_size]
            return logits
