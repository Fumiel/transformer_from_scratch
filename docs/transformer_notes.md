# Transformer Notes

## Embedding

Token ID をベクトルに変換する。

```text
input_ids: [B, T]
token_embedding(input_ids): [B, T, C]
```

## Causal Self-Attention

Decoder-only Transformer では、未来 token を見てはいけない。
そのため、attention score `[T, T]` に下三角 mask をかける。

## Q, K, V

- Q: 自分が何を探しているか
- K: 各 token が何を持っているか
- V: 実際に集める情報

## AI Security Connection

Prompt injection は、未信頼テキストがモデルの出力分布や後続の tool call に影響する問題として見られる。
まずは小さい Transformer で、入力 prompt の違いが logits にどう影響するかを観察する。
