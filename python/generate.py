from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tiny_transformer import CharTokenizer, TinyGPTConfig
from tiny_transformer.model import TinyGPT
from tiny_transformer.tracing import GenerationStepTrace, GenerationTrace, TopKToken


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
    return_trace: bool = False,
    trace_top_k: int = 5,
) -> str | tuple[str, GenerationTrace]:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if return_trace and trace_top_k < 1:
        raise ValueError("trace_top_k must be >= 1")

    prompt_token_ids = tokenizer.encode(prompt)
    token_ids = list(prompt_token_ids)
    generation_steps = []

    for generation_step in range(max_new_tokens):
        # 系列がblock_sizeを超える場合、最後のblock_sizeトークンのみを使用する。
        # つまり最初のトークンは切り捨てられる。
        context = token_ids[-config.block_size :]
        context_start = len(token_ids) - len(context)
        # [B, T]
        x = torch.tensor([context], dtype=torch.long, device=device)

        # 勾配なし推論。
        with torch.no_grad():
            model_result = model(x, return_trace=return_trace, top_k=trace_top_k)
        if return_trace:
            logits, model_trace = model_result
        else:
            logits = model_result

        # greedy decoding: 最後のトークンのlogitsから最大値を持つトークンIDを選択する。
        # 高確率な候補から確率的に選ぶsamplingと比べると再現性が高いが、生成されるテキストの多様性は低くなる。
        next_token_id = int(torch.argmax(logits[0, -1, :]).item())

        if return_trace:
            last_position_ids = model_trace.top_k_ids[0, -1]
            last_position_logits = model_trace.top_k_logits[0, -1]
            last_position_probabilities = model_trace.top_k_probabilities[0, -1]
            top_k_tokens = [
                TopKToken(
                    token_id=int(token_id.item()),
                    token=tokenizer.decode([int(token_id.item())]),
                    logit=float(token_logit.item()),
                    probability=float(token_probability.item()),
                )
                for token_id, token_logit, token_probability in zip(
                    last_position_ids,
                    last_position_logits,
                    last_position_probabilities,
                )
            ]
            selected_probability = torch.softmax(logits[0, -1, :], dim=-1)[next_token_id]
            generation_steps.append(
                GenerationStepTrace(
                    step=generation_step,
                    context_start=context_start,
                    context_token_ids=list(context),
                    context_text=tokenizer.decode(context),
                    selected_token_id=next_token_id,
                    selected_token=tokenizer.decode([next_token_id]),
                    selected_token_logit=float(logits[0, -1, next_token_id].item()),
                    selected_token_probability=float(selected_probability.item()),
                    top_k=top_k_tokens,
                    model_trace=model_trace,
                )
            )
        token_ids.append(next_token_id)

    generated_text = tokenizer.decode(token_ids)
    if not return_trace:
        return generated_text

    trace = GenerationTrace(
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        decoding_method="greedy",
        steps=generation_steps,
        generated_token_ids=list(token_ids),
        generated_text=generated_text,
    )
    return generated_text, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tiny.pt"))
    parser.add_argument("--prompt", type=str, default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--trace-top-k", type=int, default=5)
    args = parser.parse_args()

    # CUDA、MPS、CPUの順に利用可能なデバイスを選ぶ。
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model, tokenizer, config = load_checkpoint(args.checkpoint, device=device)
    generation_result = generate_text(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
        return_trace=args.trace_output is not None,
        trace_top_k=args.trace_top_k,
    )
    if args.trace_output is not None:
        generated, trace = generation_result
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trace, args.trace_output)
    else:
        generated = generation_result
    print(generated)
    if args.trace_output is not None:
        print(f"saved generation trace to {args.trace_output}")


if __name__ == "__main__":
    main()
