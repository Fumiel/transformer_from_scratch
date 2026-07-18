import json
import sys
from pathlib import Path

import pytest

from tiny_transformer import CharTokenizer, TinyGPTConfig
from tiny_transformer.model import TinyGPT
from train import (
    build_training_batch,
    main,
    resolve_probe_text,
    save_checkpoint,
    trace_capture_steps,
    train_model,
)


def test_trace_capture_steps_emphasizes_early_and_ten_later_intervals() -> None:
    assert trace_capture_steps(1000) == [1, 10, *range(100, 1001, 100)]
    assert trace_capture_steps(2) == [1, 2]


def test_resolve_probe_text_uses_deterministic_corpus_prefix() -> None:
    assert resolve_probe_text("abcdefghijklmnopqrstuvwxyz", block_size=8, probe_text=None) == "abcdefgh"
    assert resolve_probe_text("training text", block_size=16, probe_text="probe") == "probe"


def test_resolve_probe_text_rejects_empty_or_too_long_probe() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_probe_text("training text", block_size=8, probe_text="")
    with pytest.raises(ValueError, match="must not exceed block_size"):
        resolve_probe_text("training text", block_size=4, probe_text="hello")


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


def test_main_accepts_trace_cli_options_and_uses_default_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    data_path = tmp_path / "corpus.txt"
    checkpoint_path = tmp_path / "tiny.pt"
    trace_dir = tmp_path / "traces"
    text = "hello transformer training loop\n" * 8
    data_path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data",
            str(data_path),
            "--steps",
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
            "--trace-dir",
            str(trace_dir),
            "--trace-top-k",
            "3",
        ],
    )

    main()

    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    assert checkpoint_path.exists()
    assert metadata["probe_text"] == text[:32]
    assert (trace_dir / "step_000001_probe.pt").exists()


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


def test_train_model_saves_fixed_probe_traces_and_metadata(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    data_path = tmp_path / "corpus.txt"
    checkpoint_path = tmp_path / "tiny.pt"
    trace_dir = tmp_path / "traces"
    data_path.write_text("hello transformer training loop\n" * 8, encoding="utf-8")

    train_model(
        data_path=data_path,
        steps=3,
        learning_rate=1e-3,
        checkpoint_path=checkpoint_path,
        log_every=1,
        batch_size=2,
        n_layer=1,
        n_head=2,
        n_embd=8,
        trace_dir=trace_dir,
        probe_text="hello",
        trace_top_k=3,
    )

    metadata = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (trace_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    trace_paths = sorted(trace_dir.glob("step_*_probe.pt"))

    assert metadata["probe_text"] == "hello"
    assert metadata["probe_batch_size"] == 1
    assert metadata["training_batch_size"] == 2
    assert metadata["capture_steps"] == [1, 2, 3]
    assert metadata["model_state"] == "post_optimizer_step"
    assert len(metrics) == 3
    assert all(metric["trace_saved"] for metric in metrics)
    assert [path.name for path in trace_paths] == [
        "step_000001_probe.pt",
        "step_000002_probe.pt",
        "step_000003_probe.pt",
    ]

    payload = torch.load(trace_paths[-1], map_location="cpu", weights_only=False)
    assert payload["step"] == 3
    assert payload["model_state"] == "post_optimizer_step"
    assert payload["trace"].input_ids.shape[0] == 1
    assert payload["trace"].input_ids.tolist() == [metadata["probe_token_ids"]]
    assert payload["trace"].top_k_ids.shape[-1] == 3


def test_train_model_rejects_non_empty_trace_directory(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    data_path = tmp_path / "corpus.txt"
    trace_dir = tmp_path / "traces"
    data_path.write_text("abcabcabc", encoding="utf-8")
    trace_dir.mkdir()
    existing_file = trace_dir / "existing.txt"
    existing_file.write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="--trace-dir must be empty"):
        train_model(
            data_path=data_path,
            steps=1,
            learning_rate=1e-3,
            checkpoint_path=tmp_path / "tiny.pt",
            log_every=1,
            batch_size=1,
            n_layer=1,
            n_head=1,
            n_embd=4,
            trace_dir=trace_dir,
            probe_text="abc",
        )

    assert existing_file.read_text(encoding="utf-8") == "keep me"
    assert not (trace_dir / "metadata.json").exists()
    assert not (tmp_path / "tiny.pt").exists()


def test_train_model_rejects_unknown_character_in_explicit_probe(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    data_path = tmp_path / "corpus.txt"
    data_path.write_text("abcabcabc", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown character"):
        train_model(
            data_path=data_path,
            steps=1,
            learning_rate=1e-3,
            checkpoint_path=tmp_path / "tiny.pt",
            log_every=1,
            batch_size=1,
            n_layer=1,
            n_head=1,
            n_embd=4,
            trace_dir=tmp_path / "traces",
            probe_text="z",
        )
