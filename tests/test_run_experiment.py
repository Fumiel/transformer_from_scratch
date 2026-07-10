import sys
from pathlib import Path

import pytest

from run_experiment import append_generation_log, main


def test_append_generation_log_writes_metadata_and_output(tmp_path: Path) -> None:
    log_path = tmp_path / "generation_log.md"

    append_generation_log(
        log_path=log_path,
        checkpoint_path=Path("checkpoints/tiny.pt"),
        prompt="hello",
        max_new_tokens=5,
        steps=100,
        batch_size=4,
        n_layer=1,
        n_head=2,
        n_embd=8,
        learning_rate=1e-3,
        final_loss=1.234567,
        generated_text="hello world",
    )

    content = log_path.read_text(encoding="utf-8")
    assert "checkpoint: `checkpoints/tiny.pt`" in content
    assert "prompt: `hello`" in content
    assert "final_loss: `1.234567`" in content
    assert "hello world" in content


def test_main_trains_generates_and_appends_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    pytest.importorskip("torch")

    data_path = tmp_path / "corpus.txt"
    checkpoint_path = tmp_path / "tiny.pt"
    log_path = tmp_path / "generation_log.md"
    data_path.write_text("hello transformer training loop\n" * 8, encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
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
            "--prompt",
            "hello",
            "--max-new-tokens",
            "2",
            "--output-log",
            str(log_path),
        ],
    )

    main()

    output = capsys.readouterr().out
    assert checkpoint_path.exists()
    assert log_path.exists()
    assert "saved generation log to" in output
