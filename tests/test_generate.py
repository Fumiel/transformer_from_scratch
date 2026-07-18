import sys
from pathlib import Path

import pytest

from generate import generate_text, load_checkpoint, main, tokenizer_from_vocab
from tiny_transformer import TinyGPTConfig
from tiny_transformer.model import TinyGPT
from train import save_checkpoint


def zero_model_parameters(model: TinyGPT) -> None:
    for param in model.parameters():
        param.data.zero_()


def test_tokenizer_from_vocab_preserves_checkpoint_token_ids() -> None:
    tokenizer = tokenizer_from_vocab(["b", "a", "\n"])

    assert tokenizer.encode("ba\n") == [0, 1, 2]
    assert tokenizer.decode([2, 1, 0]) == "\nab"


def test_generate_text_uses_greedy_decoding() -> None:
    torch = pytest.importorskip("torch")

    tokenizer = tokenizer_from_vocab(["a", "b"])
    config = TinyGPTConfig(vocab_size=2, block_size=2, n_layer=1, n_head=1, n_embd=4)
    model = TinyGPT(config)
    zero_model_parameters(model)

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt="b",
        max_new_tokens=3,
        device=torch.device("cpu"),
    )

    assert generated == "baaa"


def test_generate_text_truncates_context_to_block_size() -> None:
    torch = pytest.importorskip("torch")

    tokenizer = tokenizer_from_vocab(["a", "b"])
    config = TinyGPTConfig(vocab_size=2, block_size=2, n_layer=1, n_head=1, n_embd=4)
    model = TinyGPT(config)
    zero_model_parameters(model)

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt="bbb",
        max_new_tokens=1,
        device=torch.device("cpu"),
    )

    assert generated == "bbba"


def test_generate_text_can_return_step_traces_without_changing_text() -> None:
    torch = pytest.importorskip("torch")

    tokenizer = tokenizer_from_vocab(["a", "b"])
    config = TinyGPTConfig(vocab_size=2, block_size=2, n_layer=1, n_head=1, n_embd=4)
    model = TinyGPT(config)
    zero_model_parameters(model)

    generated, trace = generate_text(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt="bbb",
        max_new_tokens=2,
        device=torch.device("cpu"),
        return_trace=True,
        trace_top_k=2,
    )

    assert generated == "bbbaa"
    assert trace.prompt == "bbb"
    assert trace.prompt_token_ids == [1, 1, 1]
    assert trace.decoding_method == "greedy"
    assert trace.generated_token_ids == [1, 1, 1, 0, 0]
    assert trace.generated_text == generated
    assert len(trace.steps) == 2

    first_step, second_step = trace.steps
    assert first_step.step == 0
    assert first_step.context_start == 1
    assert first_step.context_token_ids == [1, 1]
    assert first_step.context_text == "bb"
    assert first_step.selected_token_id == 0
    assert first_step.selected_token == "a"
    assert len(first_step.top_k) == 2
    assert first_step.model_trace.input_ids.tolist() == [[1, 1]]
    assert second_step.step == 1
    assert second_step.context_start == 2
    assert second_step.context_token_ids == [1, 0]
    assert second_step.context_text == "ba"


def test_generate_text_rejects_non_positive_trace_top_k() -> None:
    torch = pytest.importorskip("torch")

    tokenizer = tokenizer_from_vocab(["a", "b"])
    config = TinyGPTConfig(vocab_size=2, block_size=2, n_layer=1, n_head=1, n_embd=4)
    model = TinyGPT(config)

    with pytest.raises(ValueError, match="trace_top_k must be >= 1"):
        generate_text(
            model=model,
            tokenizer=tokenizer,
            config=config,
            prompt="b",
            max_new_tokens=1,
            device=torch.device("cpu"),
            return_trace=True,
            trace_top_k=0,
        )


def test_load_checkpoint_restores_model_config_and_vocab(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    tokenizer = tokenizer_from_vocab(["a", "b"])
    config = TinyGPTConfig(vocab_size=2, block_size=2, n_layer=1, n_head=1, n_embd=4)
    model = TinyGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "tiny.pt"

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        step=1,
        loss=0.5,
    )

    loaded_model, loaded_tokenizer, loaded_config = load_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
    )

    assert loaded_config == config
    assert loaded_tokenizer.encode("ba") == [1, 0]
    assert loaded_model.training is False


def test_main_prints_generated_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    torch = pytest.importorskip("torch")

    tokenizer = tokenizer_from_vocab(["a", "b"])
    config = TinyGPTConfig(vocab_size=2, block_size=2, n_layer=1, n_head=1, n_embd=4)
    model = TinyGPT(config)
    zero_model_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "tiny.pt"
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        step=1,
        loss=0.5,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            "b",
            "--max-new-tokens",
            "2",
        ],
    )

    main()

    assert capsys.readouterr().out == "baa\n"


def test_main_saves_generation_trace_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    torch = pytest.importorskip("torch")

    tokenizer = tokenizer_from_vocab(["a", "b"])
    config = TinyGPTConfig(vocab_size=2, block_size=2, n_layer=1, n_head=1, n_embd=4)
    model = TinyGPT(config)
    zero_model_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "tiny.pt"
    trace_path = tmp_path / "nested" / "generation_trace.pt"
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        step=1,
        loss=0.5,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            "b",
            "--max-new-tokens",
            "2",
            "--trace-output",
            str(trace_path),
            "--trace-top-k",
            "2",
        ],
    )

    main()

    output = capsys.readouterr().out
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
    assert output == f"baa\nsaved generation trace to {trace_path}\n"
    assert trace.generated_text == "baa"
    assert len(trace.steps) == 2
    assert len(trace.steps[0].top_k) == 2
