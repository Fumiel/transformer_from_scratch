import pytest

from tiny_transformer.attention import scaled_dot_product_attention
from tiny_transformer.config import TinyGPTConfig
from tiny_transformer.model import TinyGPT
from tiny_transformer.tracing import capture_tensor


def test_capture_tensor_creates_independent_cpu_snapshot() -> None:
    torch = pytest.importorskip("torch")
    source = torch.tensor([1.0, 2.0], requires_grad=True)

    snapshot = capture_tensor(source)
    with torch.no_grad():
        source.add_(10.0)

    assert snapshot.device.type == "cpu"
    assert snapshot.requires_grad is False
    assert snapshot.grad_fn is None
    assert torch.equal(snapshot, torch.tensor([1.0, 2.0]))


def test_scaled_dot_product_attention_keeps_original_return_type_by_default() -> None:
    torch = pytest.importorskip("torch")
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    mask = torch.tril(torch.ones(3, 3)).view(1, 1, 3, 3)

    output = scaled_dot_product_attention(q, k, v, mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 2, 3, 4)


def test_model_trace_has_expected_shapes_and_preserves_logits() -> None:
    torch = pytest.importorskip("torch")
    config = TinyGPTConfig(vocab_size=7, block_size=5, n_layer=2, n_head=2, n_embd=8)
    model = TinyGPT(config)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    normal_logits = model(input_ids)
    traced_logits, trace = model(input_ids, return_trace=True, top_k=3)

    assert torch.equal(traced_logits, normal_logits)
    assert trace.input_ids.shape == (1, 3)
    assert trace.token_embedding.shape == (1, 3, 8)
    assert trace.position_embedding.shape == (1, 3, 8)
    assert trace.input_embedding.shape == (1, 3, 8)
    assert len(trace.blocks) == 2
    assert trace.final_ln_output.shape == (1, 3, 8)
    assert trace.logits.shape == (1, 3, 7)
    assert trace.top_k_ids.shape == (1, 3, 3)
    assert trace.top_k_logits.shape == (1, 3, 3)
    assert trace.top_k_probabilities.shape == (1, 3, 3)


def test_block_trace_records_attention_ffn_and_residual_relationships() -> None:
    torch = pytest.importorskip("torch")
    config = TinyGPTConfig(vocab_size=7, block_size=5, n_layer=1, n_head=2, n_embd=8)
    model = TinyGPT(config)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    _, trace = model(input_ids, return_trace=True)
    block = trace.blocks[0]

    assert block.block_input.shape == (1, 3, 8)
    assert block.ln1_output.shape == (1, 3, 8)
    assert block.attention.q.shape == (1, 2, 3, 4)
    assert block.attention.k.shape == (1, 2, 3, 4)
    assert block.attention.v.shape == (1, 2, 3, 4)
    assert block.attention.scores.shape == (1, 2, 3, 3)
    assert block.attention.masked_scores.shape == (1, 2, 3, 3)
    assert block.attention.probabilities.shape == (1, 2, 3, 3)
    assert block.attention.head_output.shape == (1, 2, 3, 4)
    assert block.attention.merged_output.shape == (1, 3, 8)
    assert block.attention.output_projection.shape == (1, 3, 8)
    assert block.ffn.fc1_output.shape == (1, 3, 32)
    assert block.ffn.gelu_output.shape == (1, 3, 32)
    assert block.ffn.fc2_output.shape == (1, 3, 8)
    assert torch.equal(
        block.after_attention_residual,
        block.block_input + block.attention_residual_delta,
    )
    assert torch.equal(
        block.block_output,
        block.after_attention_residual + block.ffn_residual_delta,
    )
    assert torch.equal(block.attention_residual_delta, block.attention.output_projection)
    assert torch.equal(block.ffn_residual_delta, block.ffn.fc2_output)


def test_attention_trace_respects_causal_mask_and_probability_normalization() -> None:
    torch = pytest.importorskip("torch")
    config = TinyGPTConfig(vocab_size=7, block_size=5, n_layer=1, n_head=2, n_embd=8)
    model = TinyGPT(config)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    _, trace = model(input_ids, return_trace=True)
    attention = trace.blocks[0].attention
    future_positions = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)

    assert torch.isneginf(attention.masked_scores[..., future_positions]).all()
    assert torch.equal(
        attention.probabilities[..., future_positions],
        torch.zeros_like(attention.probabilities[..., future_positions]),
    )
    assert torch.allclose(
        attention.probabilities.sum(dim=-1),
        torch.ones(1, 2, 3),
        atol=1e-6,
    )


def test_top_k_values_match_full_logits_distribution() -> None:
    torch = pytest.importorskip("torch")
    config = TinyGPTConfig(vocab_size=4, block_size=3, n_layer=1, n_head=1, n_embd=4)
    model = TinyGPT(config)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    _, trace = model(input_ids, return_trace=True, top_k=10)
    probabilities = torch.softmax(trace.logits, dim=-1)

    assert trace.top_k_ids.shape[-1] == config.vocab_size
    assert torch.equal(
        trace.top_k_logits,
        trace.logits.gather(dim=-1, index=trace.top_k_ids),
    )
    assert torch.equal(
        trace.top_k_probabilities,
        probabilities.gather(dim=-1, index=trace.top_k_ids),
    )


def test_tracing_rejects_non_positive_top_k() -> None:
    torch = pytest.importorskip("torch")
    config = TinyGPTConfig(vocab_size=4, block_size=3, n_layer=1, n_head=1, n_embd=4)
    model = TinyGPT(config)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        model(input_ids, return_trace=True, top_k=0)
