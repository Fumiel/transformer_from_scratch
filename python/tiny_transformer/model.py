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
    from .tracing import BlockTrace, FeedForwardTrace, ModelTrace, capture_tensor
except ModuleNotFoundError:
    torch = None
    nn = object  # type: ignore[assignment]


if torch is not None:

    # [B, T, n_embd] -> [B, T, n_embd]
    class FeedForward(nn.Module): # nn.Moduleを継承して層を作る。
        def __init__(self, config: TinyGPTConfig) -> None:
            super().__init__()
            self.fc1 = Linear(config.n_embd, 4 * config.n_embd)
            self.fc2 = Linear(4 * config.n_embd, config.n_embd)

        def forward(
            self,
            x: torch.Tensor,
            return_trace: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, FeedForwardTrace]:
            fc1_output = self.fc1(x)
            gelu_output = gelu(fc1_output)
            fc2_output = self.fc2(gelu_output)

            if not return_trace:
                return fc2_output

            trace = FeedForwardTrace(
                fc1_output=capture_tensor(fc1_output),
                gelu_output=capture_tensor(gelu_output),
                fc2_output=capture_tensor(fc2_output),
            )
            return fc2_output, trace


    class TransformerBlock(nn.Module):
        def __init__(self, config: TinyGPTConfig) -> None:
            super().__init__()
            self.ln1 = LayerNorm(config.n_embd)
            self.attn = MultiHeadCausalSelfAttention(config)
            self.ln2 = LayerNorm(config.n_embd)
            self.ffn = FeedForward(config)

        def forward(
            self,
            x: torch.Tensor,
            return_trace: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, BlockTrace]:
            block_input = x
            ln1_output = self.ln1(block_input)
            attention_result = self.attn(ln1_output, return_trace=return_trace)
            if return_trace:
                attention_delta, attention_trace = attention_result
            else:
                attention_delta = attention_result
            after_attention_residual = block_input + attention_delta

            ln2_output = self.ln2(after_attention_residual)
            ffn_result = self.ffn(ln2_output, return_trace=return_trace)
            if return_trace:
                ffn_delta, ffn_trace = ffn_result
            else:
                ffn_delta = ffn_result
            block_output = after_attention_residual + ffn_delta

            if not return_trace:
                return block_output

            trace = BlockTrace(
                block_input=capture_tensor(block_input),
                ln1_output=capture_tensor(ln1_output),
                attention=attention_trace,
                attention_residual_delta=capture_tensor(attention_delta),
                after_attention_residual=capture_tensor(after_attention_residual),
                ln2_output=capture_tensor(ln2_output),
                ffn=ffn_trace,
                ffn_residual_delta=capture_tensor(ffn_delta),
                block_output=capture_tensor(block_output),
            )
            return block_output, trace


    class TinyGPT(nn.Module):
        def __init__(self, config: TinyGPTConfig) -> None:
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
            self.final_ln = LayerNorm(config.n_embd)
            self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

        def forward(
            self,
            input_ids: torch.Tensor,
            return_trace: bool = False,
            top_k: int = 5,
        ) -> torch.Tensor | tuple[torch.Tensor, ModelTrace]:
            # input_ids: [B, T]
            bsz, seq_len = input_ids.shape
            if seq_len > self.config.block_size:
                raise ValueError("sequence length exceeds block_size")
            if return_trace and top_k < 1:
                raise ValueError("top_k must be >= 1")

            positions = torch.arange(seq_len, device=input_ids.device)  # [T]
            # [B, T, C] + [1, T, C]
            token_embedding = self.token_embedding(input_ids)
            position_embedding = self.position_embedding(positions)[None, :, :]
            input_embedding = token_embedding + position_embedding
            x = input_embedding
            block_traces = []
            for block in self.blocks:
                block_result = block(x, return_trace=return_trace)
                if return_trace:
                    x, block_trace = block_result
                    block_traces.append(block_trace)
                else:
                    x = block_result
            final_ln_output = self.final_ln(x) # [B, T, C]
            logits = self.lm_head(final_ln_output)  # [B, T, vocab_size]

            if not return_trace:
                return logits

            # 指定値top_kに対して、vocab_sizeを超えないようにする。
            effective_top_k = min(top_k, self.config.vocab_size)
            # 上位候補のlogitsとそのインデックスを取得する。
            top_k_logits, top_k_ids = torch.topk(logits, k=effective_top_k, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            top_k_probabilities = probabilities.gather(dim=-1, index=top_k_ids)
            trace = ModelTrace(
                input_ids=capture_tensor(input_ids),
                token_embedding=capture_tensor(token_embedding),
                position_embedding=capture_tensor(position_embedding),
                input_embedding=capture_tensor(input_embedding),
                blocks=block_traces,
                final_ln_output=capture_tensor(final_ln_output),
                logits=capture_tensor(logits),
                top_k_ids=capture_tensor(top_k_ids),
                top_k_logits=capture_tensor(top_k_logits),
                top_k_probabilities=capture_tensor(top_k_probabilities),
            )
            return logits, trace
