# Shape Table

| 変数 | Shape | 意味 |
|---|---:|---|
| `input_ids` | `[B, T]` | token IDs |
| `x` | `[B, T, C]` | hidden states |
| `q, k, v` | `[B, T, C]` | q/k/v before split |
| `q_head` | `[B, H, T, D]` | q after head split |
| `scores` | `[B, H, T, T]` | q @ k^T |
| `probs` | `[B, H, T, T]` | softmax scores |
| `out` | `[B, H, T, D]` | probs @ v |
| `merged` | `[B, T, C]` | merged heads |
| `logits` | `[B, T, V]` | next-token logits |

```text
B: batch size
T: sequence length
C: embedding dimension
H: number of heads
D: head dimension
V: vocab size
```
