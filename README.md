# Transformer from Scratch: Python Training + C++ Inference

櫻井研(2026)でセキュリティfor AIエージェントの研究を実施するための実装力強化・AI基礎理論の理解を目標として取り組む。現代のLLM研究・開発において重要概念であるTransformer の内部構造を理解するための実装演習リポジトリです。  
Python では小さな Decoder-only Transformer を学習し、C++ では export された重みを読み込んで同じ forward 推論を再現します。

うまくいったら(最終的には、この実装を AI セキュリティ実験、特に prompt injection、activation/logit 解析、guardrail 評価、LLM agent security の基盤として使います。)

---

## Goal

このリポジトリの最初のゴールは、以下を参加者が自力で説明・実装できるようになることです。

```text
PythonでTiny Transformerを実装・学習する
  ↓
学習済み重みを保存/exportする
  ↓
C++で同じTransformerの推論器を実装する
  ↓
Python版とC++版のlogits差分を比較する
```

到達目標：

- Transformer の forward pass を shape 付きで追える
- Causal Self-Attention を実装できる
- PyTorch の低レベル演算で学習ループを書ける
- C++ で Tensor、Linear、LayerNorm、Softmax、Attention を実装できる
- Python/C++ の数値差分をテストできる
- 実装結果を README、実験レポート、PR で説明できる

---

## Why this project?

AI セキュリティ研究では、LLM を単に API として使うだけでは不十分です。  
Prompt injection、jailbreak、tool-use agent、guardrail、activation monitoring などを扱うには、少なくとも次の感覚が必要になります。

- 入力 token がどのように embedding されるか
- attention がどの token 間の情報伝播を作るか
- logits がどう次 token の分布になるか
- 数値誤差、mask、softmax、LayerNorm が挙動にどう影響するか
- モデルを Python 以外の環境でどう再現するか

このプロジェクトでは、Tiny GPT 風の Decoder-only Transformer を自作することで、その基礎を固めます。

---

## Architecture

実装するモデルは最小の Decoder-only Transformer です。

```text
input token ids
  ↓
Token Embedding + Position Embedding
  ↓
Transformer Block × N
  ├─ LayerNorm
  ├─ Multi-Head Causal Self-Attention
  ├─ Residual Connection
  ├─ LayerNorm
  ├─ Feed Forward Network
  └─ Residual Connection
  ↓
Final LayerNorm
  ↓
LM Head
  ↓
next-token logits
```

初期モデル設定：

```python
vocab_size = 128
block_size = 64
n_layer = 2
n_head = 2
n_embd = 64
dropout = 0.0
```

C++ 推論との比較を簡単にするため、最初は dropout を 0 にします。

---

## Shape Table

Transformer 実装で最重要なのは shape です。

| 変数 | Shape | 意味 |
|---|---:|---|
| `input_ids` | `[B, T]` | token ID列 |
| `token_emb` | `[B, T, C]` | token embedding |
| `pos_emb` | `[T, C]` | position embedding |
| `x` | `[B, T, C]` | block入力 |
| `q, k, v` | `[B, T, C]` | query/key/value |
| `q reshaped` | `[B, H, T, D]` | headごとに分割後 |
| `attn_scores` | `[B, H, T, T]` | attention score |
| `attn_probs` | `[B, H, T, T]` | softmax後 |
| `attn_out` | `[B, H, T, D]` | value集約後 |
| `merged` | `[B, T, C]` | head結合後 |
| `logits` | `[B, T, vocab_size]` | 次token予測 |

記号：

```text
B = batch size
T = sequence length
C = embedding dimension
H = number of heads
D = head dimension = C / H
```

---

## Repository Structure

```text
transformer-from-scratch/
  ├─ README.md
  ├─ pyproject.toml
  ├─ CMakeLists.txt
  ├─ Makefile
  ├─ data/
  │   └─ tiny_corpus.txt
  │
  ├─ python/
  │   ├─ train.py
  │   ├─ generate.py
  │   ├─ export_weights.py
  │   ├─ compare_cpp.py
  │   └─ tiny_transformer/
  │       ├─ __init__.py
  │       ├─ tokenizer.py
  │       ├─ config.py
  │       ├─ layers.py
  │       ├─ attention.py
  │       └─ model.py
  │
  ├─ cpp/
  │   ├─ main.cpp
  │   ├─ tensor.hpp
  │   ├─ tensor.cpp
  │   ├─ layers.hpp
  │   ├─ layers.cpp
  │   ├─ attention.hpp
  │   ├─ attention.cpp
  │   ├─ model.hpp
  │   ├─ model.cpp
  │   └─ weights_loader.cpp
  │
  ├─ tests/
  │   └─ test_tokenizer.py
  │
  ├─ docs/
  │   ├─ transformer_notes.md
  │   ├─ shape_table.md
  │   ├─ experiment_report.md
  │   └─ initial_issues.md
  │
  └─ .github/
      └─ workflows/
          └─ ci.yml
```

---

## Setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

PyTorch は環境に応じて公式手順で入れてください。CPU だけなら例として以下です。

```bash
pip install torch
```

### C++

```bash
cmake -S . -B build
cmake --build build
./build/tiny_transformer_cpp
```

---

## Usage

### Python tokenizer test

```bash
pytest
```

### Python training

まだ初期状態では TODO 実装が残っています。実装が進んだら以下で学習します。

```bash
python python/train.py --data data/tiny_corpus.txt
```

### Text generation

```bash
python python/generate.py --prompt "hello"
```

### Export weights

```bash
python python/export_weights.py --checkpoint checkpoints/tiny.pt --out weights/tiny_weights.json
```

### C++ inference

```bash
./build/tiny_transformer_cpp --weights weights/tiny_weights.json --prompt "hello"
```

### Compare Python and C++ logits

```bash
python python/compare_cpp.py \
  --checkpoint checkpoints/tiny.pt \
  --cpp-output outputs/cpp_logits.txt
```

目標は以下です。

```text
max_abs_diff < 1e-3
```

---

## Implementation Rules

学習効果を高めるため、最初は以下を禁止します。

```text
禁止:
  - torch.nn.Transformer
  - torch.nn.MultiheadAttention
  - Hugging Face Transformers
  - Trainer系ライブラリ
```

使ってよいもの：

```text
許可:
  - torch.Tensor
  - torch.matmul
  - torch.softmax
  - torch.nn.Parameter
  - torch.optim.AdamW
  - pytest
```

C++ 側は、最初は高速化を狙わず、愚直な for loop で実装します。

---

## Milestones

### Milestone 1: Python basic components

- [ ] char-level tokenizer
- [ ] Linear
- [ ] LayerNorm
- [ ] GELU
- [ ] causal mask
- [ ] scaled dot-product attention

### Milestone 2: Python TinyGPT

- [ ] Multi-head causal self-attention
- [ ] TransformerBlock
- [ ] TinyGPT model
- [ ] training loop
- [ ] generate.py

### Milestone 3: Weight export

- [ ] checkpoint format
- [ ] JSON or binary export
- [ ] golden input/output作成

### Milestone 4: C++ inference

- [ ] Tensor class
- [ ] Linear
- [ ] LayerNorm
- [ ] GELU
- [ ] Softmax
- [ ] Attention
- [ ] TransformerBlock
- [ ] full forward

### Milestone 5: Numerical consistency

- [ ] Python logits保存
- [ ] C++ logits保存
- [ ] max absolute difference計算
- [ ] `max_abs_diff < 1e-3` を目指す

### Milestone 6: AI security bridge

- [ ] prompt injection風のtoy promptを作る
- [ ] attention/logitsの変化を見る
- [ ] 実験レポートを書く

---

## First Issues

最初の Issue は `docs/initial_issues.md` を参照してください。

基本方針：

```text
1 issue = 1 small component
1 PR = 1 meaningful change
必ず test or 実行ログを貼る
```

PR には以下を含めます。

```text
## Summary
何を実装したか

## Test
どう確認したか

## Notes
shape、数値誤差、設計判断など
```

---

## Coding Style

### Python

- type hint をできる範囲で書く
- 関数は小さく保つ
- shape をコメントで書く
- magic number を避け、`config.py` に寄せる

### C++

- C++17 を使う
- まずは可読性優先
- Tensor は row-major 前提
- assert で shape mismatch を早めに落とす
- 最適化は correctness が取れてから行う

---

## Definition of Done

このプロジェクトの最初の完成条件：

- [ ] PythonでTiny Transformerを学習できる
- [ ] 文字レベルで文章生成できる
- [ ] 学習済み重みをexportできる
- [ ] C++で同じモデルのforwardができる
- [ ] Python/C++のlogits差分を比較できる
- [ ] READMEに実行方法と実験結果がある
- [ ] docsにshape表と実験レポートがある

---

## Future Direction: AI Security

この実装が完成したら、次の AI セキュリティ実験へ進みます。

- prompt injection toy benchmark
- jailbreak prompt と normal prompt の logit 差分分析
- attention map visualization
- activation monitoring
- small guardrail classifier
- tool-use LLM agent security playground
- source-sink policy による tool-call guard

---

## License

MIT License を予定。研究室方針に合わせて変更してください。