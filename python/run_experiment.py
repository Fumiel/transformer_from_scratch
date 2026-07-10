from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from generate import generate_text, load_checkpoint
from train import train_model


def append_generation_log(
    log_path: Path,
    checkpoint_path: Path,
    prompt: str,
    max_new_tokens: int,
    steps: int,
    batch_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    learning_rate: float,
    final_loss: float,
    generated_text: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"## {timestamp}\n"
        f"- checkpoint: `{checkpoint_path}`\n"
        f"- prompt: `{prompt}`\n"
        f"- max_new_tokens: `{max_new_tokens}`\n"
        f"- steps: `{steps}`\n"
        f"- batch_size: `{batch_size}`\n"
        f"- n_layer: `{n_layer}`\n"
        f"- n_head: `{n_head}`\n"
        f"- n_embd: `{n_embd}`\n"
        f"- learning_rate: `{learning_rate}`\n"
        f"- final_loss: `{final_loss:.6f}`\n"
        "\n"
        "```text\n"
        f"{generated_text}\n"
        "```\n\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)


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
    parser.add_argument("--prompt", type=str, default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--output-log", type=Path, default=Path("outputs/generation_log.md"))
    args = parser.parse_args()

    checkpoint_path, final_loss = train_model(
        data_path=args.data,
        steps=args.steps,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint,
        log_every=args.log_every,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_checkpoint(checkpoint_path, device=device)
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    append_generation_log(
        log_path=args.output_log,
        checkpoint_path=checkpoint_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        batch_size=args.batch_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        learning_rate=args.learning_rate,
        final_loss=final_loss,
        generated_text=generated,
    )
    print(generated)
    print(f"saved generation log to {args.output_log}")


if __name__ == "__main__":
    main()
