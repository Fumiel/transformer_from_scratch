from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_transformer import CharTokenizer, TinyGPTConfig
from tiny_transformer.model import TinyGPT


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

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
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    text = load_text(args.data)
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
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    # GPUがある場合はcudaを選ぶ。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        # 毎回ランダムにバッチを作る。
        x, y = build_training_batch(
            token_ids,
            block_size=config.block_size,
            batch_size=args.batch_size,
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
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step={step} loss={last_loss:.6f}")

    save_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        step=args.steps,
        loss=last_loss,
    )
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
