"""Data structures used to capture a TinyGPT forward pass."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def capture_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return an independent CPU snapshot without retaining the computation graph."""
    return tensor.detach().to(device="cpu").clone()


@dataclass
class AttentionTrace:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    scores: torch.Tensor
    masked_scores: torch.Tensor
    probabilities: torch.Tensor
    head_output: torch.Tensor
    merged_output: torch.Tensor
    output_projection: torch.Tensor


@dataclass
class FeedForwardTrace:
    fc1_output: torch.Tensor
    gelu_output: torch.Tensor
    fc2_output: torch.Tensor


@dataclass
class BlockTrace:
    block_input: torch.Tensor
    ln1_output: torch.Tensor
    attention: AttentionTrace
    attention_residual_delta: torch.Tensor
    after_attention_residual: torch.Tensor
    ln2_output: torch.Tensor
    ffn: FeedForwardTrace
    ffn_residual_delta: torch.Tensor
    block_output: torch.Tensor


@dataclass
class ModelTrace:
    input_ids: torch.Tensor
    token_embedding: torch.Tensor
    position_embedding: torch.Tensor
    input_embedding: torch.Tensor
    blocks: list[BlockTrace]
    final_ln_output: torch.Tensor
    logits: torch.Tensor
    top_k_ids: torch.Tensor
    top_k_logits: torch.Tensor
    top_k_probabilities: torch.Tensor


@dataclass(frozen=True)
class TopKToken:
    token_id: int
    token: str
    logit: float
    probability: float


@dataclass
class GenerationStepTrace:
    step: int
    context_start: int
    context_token_ids: list[int]
    context_text: str
    selected_token_id: int
    selected_token: str
    selected_token_logit: float
    selected_token_probability: float
    top_k: list[TopKToken]
    model_trace: ModelTrace


@dataclass
class GenerationTrace:
    prompt: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    decoding_method: str
    steps: list[GenerationStepTrace]
    generated_token_ids: list[int]
    generated_text: str
