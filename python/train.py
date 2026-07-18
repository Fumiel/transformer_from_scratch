from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_transformer import CharTokenizer, TinyGPTConfig
from tiny_transformer.model import TinyGPT
from tiny_transformer.tracing import ModelTrace


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def trace_capture_steps(total_steps: int) -> list[int]:
    """Choose probe capture steps that emphasize early learning and ten later intervals."""
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")

    interval = max(1, total_steps // 10)
    steps = {1, min(10, total_steps), total_steps}
    steps.update(range(interval, total_steps + 1, interval))
    return sorted(steps)


def resolve_probe_text(text: str, block_size: int, probe_text: str | None) -> str:
    """Return an explicit probe or a deterministic prefix of the training corpus."""
    resolved = probe_text if probe_text is not None else text[: min(32, block_size)]
    if not resolved:
        raise ValueError("--probe-text must not be empty")
    if len(resolved) > block_size:
        raise ValueError("--probe-text must not exceed block_size")
    return resolved


def initialize_training_trace(
    trace_dir: Path,
    data_path: Path,
    checkpoint_path: Path,
    config: TinyGPTConfig,
    device: torch.device,
    total_steps: int,
    learning_rate: float,
    batch_size: int,
    probe_text: str,
    probe_token_ids: list[int],
    trace_top_k: int,
    capture_steps: list[int],
) -> None:
    if trace_dir.exists() and any(trace_dir.iterdir()):
        raise ValueError(f"--trace-dir must be empty: {trace_dir}")
    trace_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "checkpoint_path": str(checkpoint_path),
        "total_steps": total_steps,
        "learning_rate": learning_rate,
        "training_batch_size": batch_size,
        "device": str(device),
        "torch_version": torch.__version__,
        "config": asdict(config),
        "probe_text": probe_text,
        "probe_token_ids": probe_token_ids,
        "probe_batch_size": 1,
        "trace_top_k": trace_top_k,
        "capture_steps": capture_steps,
        "model_state": "post_optimizer_step",
    }
    (trace_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (trace_dir / "metrics.jsonl").write_text("", encoding="utf-8")


def save_probe_trace(
    trace_dir: Path,
    step: int,
    loss: float,
    trace: ModelTrace,
) -> Path:
    trace_path = trace_dir / f"step_{step:06d}_probe.pt"
    torch.save(
        {
            "step": step,
            "loss": loss,
            "model_state": "post_optimizer_step",
            "trace": trace,
        },
        trace_path,
    )
    return trace_path


def append_training_metric(
    trace_dir: Path,
    step: int,
    loss: float,
    trace_path: Path | None,
) -> None:
    metric = {
        "step": step,
        "loss": loss,
        "trace_saved": trace_path is not None,
        "trace_file": trace_path.name if trace_path is not None else None,
    }
    with (trace_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(metric, ensure_ascii=False) + "\n")

# ランダムな開始位置から長さblock_sizeの連続するトークン列を、token_idsからbatch_size個取り出す。
def build_training_batch(
    token_ids: list[int],
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) <= block_size:
        raise ValueError("training data must contain at least 2 tokens")

    max_start = len(token_ids) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))

    # xは入力トークン列、yは正解トークン列。1文字ずらすことで、次の文字を予測するタスクにする。
    x = torch.stack(
        [
            torch.tensor(token_ids[start : start + block_size], dtype=torch.long)
            for start in starts.tolist()
        ]
    ).to(device)
    y = torch.stack(
        [
            torch.tensor(token_ids[start + 1 : start + block_size + 1], dtype=torch.long)
            for start in starts.tolist()
        ]
    ).to(device)
    return x, y


def save_checkpoint(
    checkpoint_path: Path,
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    config: TinyGPTConfig,
    tokenizer: CharTokenizer,
    step: int,
    loss: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    vocab = [tokenizer.itos[i] for i in range(tokenizer.vocab_size)]
    torch.save(
        {
            "step": step,
            "loss": loss,
            "config": {
                "vocab_size": config.vocab_size,
                "block_size": config.block_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "dropout": config.dropout,
            },
            # tokenizerの語彙を保存する。これがないと、学習済みモデルをロードしても、文字列に変換できない。
            "vocab": vocab,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def train_model(
    data_path: Path,
    steps: int,
    learning_rate: float,
    checkpoint_path: Path,
    log_every: int,
    batch_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    trace_dir: Path | None = None,
    probe_text: str | None = None,
    trace_top_k: int = 5,
) -> tuple[Path, float]:
    if steps < 1:
        raise ValueError("--steps must be >= 1")
    if log_every < 1:
        raise ValueError("--log-every must be >= 1")
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if trace_dir is not None and trace_top_k < 1:
        raise ValueError("--trace-top-k must be >= 1")

    text = load_text(data_path)
    tokenizer = CharTokenizer(text) # 学習データに出てくる文字のみを語彙とする。
    token_ids = tokenizer.encode(text) # 全文を整数ID列へ。

    # 正解トークン列は入力トークン列の1文字後ろにあるので、元データ長-1が、
    # 1回に扱える最大系列長(block_size)の上限になる。
    block_size = min(64, len(token_ids) - 1)
    if block_size < 1:
        raise ValueError("training data must contain at least 2 tokens")

    config = TinyGPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )

    # CUDA、MPS、CPUの順に利用可能なデバイスを選ぶ。
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    probe_input_ids = None
    capture_steps: set[int] = set()
    if trace_dir is not None:
        resolved_probe_text = resolve_probe_text(text, config.block_size, probe_text)
        probe_token_ids = tokenizer.encode(resolved_probe_text)
        # shape = [1, T]
        # 常にB=1
        probe_input_ids = torch.tensor(
            [probe_token_ids],
            dtype=torch.long,
            device=device,
        )
        selected_capture_steps = trace_capture_steps(steps)
        capture_steps = set(selected_capture_steps)
        initialize_training_trace(
            trace_dir=trace_dir,
            data_path=data_path,
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
            total_steps=steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            probe_text=resolved_probe_text,
            probe_token_ids=probe_token_ids,
            trace_top_k=trace_top_k,
            capture_steps=selected_capture_steps,
        )

    model.train()
    last_loss = 0.0
    for step in range(1, steps + 1):
        # 毎回ランダムにバッチを作る。
        x, y = build_training_batch(
            token_ids,
            block_size=config.block_size,
            batch_size=batch_size,
            device=device,
        )
        optimizer.zero_grad()
        logits = model(x) # [B, T] -> [B, T, vocab_size]
        # F.cross_entropyは内部でsoftmaxを計算するので、logitsをsoftmaxに通す必要はない。
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
        # loss -> cross entropy -> lm_head -> TransformerBlock ->
        # attention, ffn -> embeddingの順に各パラメータについて勾配を計算。
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        trace_path = None
        if trace_dir is not None and step in capture_steps:
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    _, probe_trace = model(
                        probe_input_ids,
                        return_trace=True,
                        top_k=trace_top_k,
                    )
            finally:
                model.train(was_training)
            trace_path = save_probe_trace(
                trace_dir=trace_dir,
                step=step,
                loss=last_loss,
                trace=probe_trace,
            )
        if trace_dir is not None:
            append_training_metric(
                trace_dir=trace_dir,
                step=step,
                loss=last_loss,
                trace_path=trace_path,
            )
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"step={step} loss={last_loss:.6f}")

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        step=steps,
        loss=last_loss,
    )
    print(f"saved checkpoint to {checkpoint_path}")
    if trace_dir is not None:
        print(f"saved training traces to {trace_dir}")
    return checkpoint_path, last_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/tiny_corpus.txt"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tiny.pt"))
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--probe-text", type=str)
    parser.add_argument("--trace-top-k", type=int, default=5)
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        steps=args.steps,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint,
        log_every=args.log_every,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        trace_dir=args.trace_dir,
        probe_text=args.probe_text,
        trace_top_k=args.trace_top_k,
    )


if __name__ == "__main__":
    main()
