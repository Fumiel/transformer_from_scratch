from dataclasses import dataclass


@dataclass(frozen=True)
class TinyGPTConfig:
    vocab_size: int = 128 # (語彙として)扱えるトークン種類数。
    block_size: int = 64 # 1回に扱える最大系列長(入力文のトークンの最大個数)。
    n_layer: int = 2 # Transformerブロックの数。
    n_head: int = 2 # Multi-Head Attentionのヘッド数。
    n_embd: int = 64 # トークン埋め込み(整数ID->多次元ベクトル)の次元数。
    dropout: float = 0.0 # 実装確認優先のためランダム性は排除。

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
