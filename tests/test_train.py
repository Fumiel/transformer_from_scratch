import sys
from pathlib import Path

import pytest

from tiny_transformer import CharTokenizer, TinyGPTConfig
from tiny_transformer.model import TinyGPT
from train import build_training_batch, main, save_checkpoint, train_model


def test_build_training_batch_shifts_targets_by_one_token(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.tensor([0, 2]))
    token_ids = [10, 11, 12, 13, 14, 15]

    x, y = build_training_batch(
        token_ids=token_ids,
        block_size=3,
        batch_size=2,
        device=torch.device("cpu"),
    )

    expected_x = torch.tensor([[10, 11, 12], [12, 13, 14]])
    expected_y = torch.tensor([[11, 12, 13], [13, 14, 15]])
    assert torch.equal(x.cpu(), expected_x)
    assert torch.equal(y.cpu(), expected_y)


def test_build_training_batch_rejects_too_short_data() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="training data must contain at least 2 tokens"):
        build_training_batch(
            token_ids=[1, 2, 3],
            block_size=3,
            batch_size=1,
            device=torch.device("cpu"),
        )


def test_save_checkpoint_stores_training_state(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    tokenizer = CharTokenizer("abcab")
    config = TinyGPTConfig(vocab_size=tokenizer.vocab_size, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = TinyGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "tiny.pt"

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        step=7,
        loss=1.25,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["step"] == 7
    assert checkpoint["loss"] == 1.25
    assert checkpoint["config"]["vocab_size"] == tokenizer.vocab_size
    assert checkpoint["vocab"] == ["a", "b", "c"]
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_main_runs_training_and_writes_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")

    data_path = tmp_path / "corpus.txt"
    checkpoint_path = tmp_path / "tiny.pt"
    data_path.write_text("hello transformer training loop\n" * 8, encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data",
            str(data_path),
            "--steps",
            "2",
            "--log-every",
            "1",
            "--batch-size",
            "2",
            "--n-layer",
            "1",
            "--n-head",
            "2",
            "--n-embd",
            "8",
            "--checkpoint",
            str(checkpoint_path),
        ],
    )

    main()

    assert checkpoint_path.exists()


def test_train_model_returns_checkpoint_path_and_loss(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    data_path = tmp_path / "corpus.txt"
    checkpoint_path = tmp_path / "tiny.pt"
    data_path.write_text("hello transformer training loop\n" * 8, encoding="utf-8")

    returned_checkpoint, final_loss = train_model(
        data_path=data_path,
        steps=2,
        learning_rate=1e-3,
        checkpoint_path=checkpoint_path,
        log_every=1,
        batch_size=2,
        n_layer=1,
        n_head=2,
        n_embd=8,
    )

    assert returned_checkpoint == checkpoint_path
    assert checkpoint_path.exists()
    assert isinstance(final_loss, float)
