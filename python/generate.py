from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tiny_transformer import CharTokenizer, TinyGPTConfig
from tiny_transformer.model import TinyGPT


def tokenizer_from_vocab(vocab: list[str]) -> CharTokenizer:
    # 同じ文字が2回語彙に含まれている場合はエラーを発生させる。
    if len(set(vocab)) != len(vocab):
        raise ValueError("checkpoint vocab contains duplicate tokens")

    tokenizer = CharTokenizer("")
    tokenizer.stoi = {ch: i for i, ch in enumerate(vocab)}
    tokenizer.itos = {i: ch for ch, i in tokenizer.stoi.items()}
    return tokenizer


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TinyGPT, CharTokenizer, TinyGPTConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TinyGPTConfig(**checkpoint["config"])
    tokenizer = tokenizer_from_vocab(checkpoint["vocab"])

    model = TinyGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # eval()を呼び出すことで、モデルを推論モードに設定する。
    # これにより、ドロップアウトやバッチ正規化などの挙動が推論時のものに切り替わる。
    model.eval()
    return model, tokenizer, config


def generate_text(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    config: TinyGPTConfig,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    token_ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        # 系列がblock_sizeを超える場合、最後のblock_sizeトークンのみを使用する。
        # つまり最初のトークンは切り捨てられる。
        context = token_ids[-config.block_size :]
        # [B, T]
        x = torch.tensor([context], dtype=torch.long, device=device)

        # 勾配なし推論。
        with torch.no_grad():
            logits = model(x) # [B, T, vocab_size]

        # greedy decoding: 最後のトークンのlogitsから最大値を持つトークンIDを選択する。
        # 高確率な候補から確率的に選ぶsamplingと比べると再現性が高いが、生成されるテキストの多様性は低くなる。
        next_token_id = int(torch.argmax(logits[0, -1, :]).item())
        token_ids.append(next_token_id)

    return tokenizer.decode(token_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tiny.pt"))
    parser.add_argument("--prompt", type=str, default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_checkpoint(args.checkpoint, device=device)
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    print(generated)


if __name__ == "__main__":
    main()
