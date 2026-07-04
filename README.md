# Transformer from Scratch

Pythonで小さなDecoder-only Transformerを学習し、その重みを使ってC++で同等の推論器を再現する個人実装プロジェクトです。
目的は、Transformerの内部構造を実装レベルで理解しつつ、将来的にAIエージェントセキュリティ実験へ接続できる最小基盤を自分の手で作ることです。

現時点では、Python側の基本部品と最小モデル構造を先に固め、C++側はforward再現のための土台を整えている段階です。

---

## Project Overview

このリポジトリでは、次の流れを一貫して実装します。

```text
PythonでTiny Transformerを実装する
  ↓
小さなコーパスで学習する
  ↓
学習済み重みをexportする
  ↓
C++で同じ構造の推論器を実装する
  ↓
PythonとC++のlogits差分を比較する
```

最終的には、この実装を次のような観察・検証の土台として使う想定です。

- promptごとの差分観察
- attentionやlogitsの挙動確認
- activation / logit解析
- AI agent securityまわりの小規模実験

---

## Why This Project

AIセキュリティ研究では、LLMを単にAPIとして使うだけでは見えない部分があります。
prompt injectionやguardrail、tool use、activation monitoringのようなテーマを扱うには、少なくとも以下を自分で追える必要があります。

- tokenがどのようにembeddingされるか
- attentionがどのtoken間に情報伝播を作るか
- logitsがどのように次token分布になるか
- mask、softmax、LayerNorm、数値誤差が挙動にどう効くか
- Python以外の実装で同じ計算をどこまで再現できるか

このプロジェクトは、その理解を抽象論ではなく実装で固めるためのものです。

---

## Current Scope

対象は最小のDecoder-only Transformerです。

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

初期設定は以下です。

```python
vocab_size = 128
block_size = 64
n_layer = 2
n_head = 2
n_embd = 64
dropout = 0.0
```

C++との比較を優先するため、最初はdropoutを使いません。

---

## Current Status

2026年7月時点の実装状況です。

### Python

- `CharTokenizer` は実装済み
- `Linear` / `LayerNorm` / `gelu` は実装済み
- causal mask と scaled dot-product attention は実装済み
- `MultiHeadCausalSelfAttention` は実装済み
- `TransformerBlock` と `TinyGPT` のforwardは実装済み
- `train.py` は未実装
- `generate.py` は未実装
- `export_weights.py` は未実装
- `compare_cpp.py` は未実装

### C++

- `Tensor` / layer / modelの骨組みは配置済み
- `main.cpp` にはsmoke test相当の入口がある
- `TinyGPTCpp::forward` はまだゼロlogitsを返す仮実装
- attention / softmax / weights loader は未実装

### Tests

- tokenizerのpytestは存在
- `FeedForward` / `TransformerBlock` / `TinyGPT` のpytestを追加済み
- model系テストは `torch` が入っていない環境ではskipされる
- C++再現性の検証はこれから追加

---

## Development Policy

このプロジェクトでは、当面は次の方針で進めます。

- まずはcorrectnessを優先する
- shapeが追える実装を保つ
- PythonとC++で同じ計算を再現できることを重視する
- 高速化や最適化は後回しにする
- ブラックボックス化しやすい高レベル実装は避ける

現時点で意図的に使わないもの:

- `torch.nn.Transformer`
- `torch.nn.MultiheadAttention`
- Hugging Face Transformers
- Trainer系ライブラリ

使うもの:

- `torch.Tensor`
- `torch.matmul`
- `torch.softmax`
- `torch.nn.Parameter`
- `torch.optim.AdamW`
- `pytest`

---

## Shape Table

Transformer実装ではshape管理を最重要事項として扱います。

| variable | shape | meaning |
|---|---:|---|
| `input_ids` | `[B, T]` | token ID列 |
| `token_emb` | `[B, T, C]` | token embedding |
| `pos_emb` | `[T, C]` | position embedding |
| `x` | `[B, T, C]` | block入力 |
| `q, k, v` | `[B, T, C]` | query / key / value |
| `q reshaped` | `[B, H, T, D]` | head分割後 |
| `attn_scores` | `[B, H, T, T]` | attention score |
| `attn_probs` | `[B, H, T, T]` | softmax後 |
| `attn_out` | `[B, H, T, D]` | value集約後 |
| `merged` | `[B, T, C]` | head結合後 |
| `logits` | `[B, T, vocab_size]` | 次token予測 |

```text
B = batch size：何個の系列を同時に処理するか
T = sequence length：１つの系列の中にトークンが何個並んでいるか。T_max = block_size。
C = embedding dimension：1トークンを何個の実数で表すか
H = number of heads
D = head dimension = C / H
```

---

## Repository Structure

```text
transformer_from_scratch/
  ├─ README.md
  ├─ pyproject.toml
  ├─ CMakeLists.txt
  ├─ Makefile
  ├─ data/
  │   └─ tiny_corpus.txt
  ├─ python/
  │   ├─ train.py
  │   ├─ generate.py
  │   ├─ export_weights.py
  │   ├─ compare_cpp.py
  │   └─ tiny_transformer/
  │       ├─ tokenizer.py
  │       ├─ config.py
  │       ├─ layers.py
  │       ├─ attention.py
  │       └─ model.py
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
  ├─ tests/
  │   ├─ test_tokenizer.py
  │   ├─ test_feedforward.py
  │   ├─ test_transformer_block.py
  │   └─ test_tiny_gpt.py
  └─ docs/
      ├─ transformer_notes.md
      ├─ shape_table.md
      ├─ experiment_report.md
      ├─ initial_issues.md
      └─ lessons/
```

`docs/lessons/` には、過去の教材・演習用メモを残しています。本READMEの主目的ではありません。

---

## Setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
pip install torch
```

### C++

```bash
cmake -S . -B build
cmake --build build
./build/tiny_transformer_cpp --prompt "hello"
```

---

## Usage

### Run tests

```bash
pytest
```

PyTorchが入っている場合はtokenizerテストに加えてmodel系テストも実行されます。
PyTorchが入っていない場合は `pytest.importorskip("torch")` によりmodel系テストはskipされます。

### Python training

現時点ではCLIだけあり、学習ループ本体は未実装です。

```bash
python python/train.py --data data/tiny_corpus.txt --steps 1000
```

### Text generation

現時点ではCLIだけあり、生成本体は未実装です。

```bash
python python/generate.py --prompt "hello" --max-new-tokens 100
```

### Export weights

現時点ではCLIだけあり、export本体は未実装です。

```bash
python python/export_weights.py --checkpoint checkpoints/tiny.pt --out weights/tiny_weights.json
```

### Compare Python and C++ logits

現時点では比較処理は未実装です。

```bash
python python/compare_cpp.py --checkpoint checkpoints/tiny.pt --cpp-output outputs/cpp_logits.txt
```

### C++ smoke test

現在のC++バイナリはpromptを受け取り、仮のlogits shapeを返すところまでです。

```bash
./build/tiny_transformer_cpp --prompt "hello"
```

---

## Planned Milestones

### Phase 1: Python training path

- training loopを実装する
- lossが下がる最小学習条件を作る
- 文字レベル生成まで通す

### Phase 2: Weight export

- checkpoint形式を決める
- Python重みのexport処理を実装する
- comparison用のgolden input / outputを保存する

### Phase 3: C++ forward reproduction

- Tensor演算を固める
- softmax / attention / block forwardを実装する
- Pythonと同じ構造のforwardを通す

### Phase 4: Numerical comparison

- Python logitsを保存する
- C++ logitsを保存する
- `max_abs_diff` を計測する
- まずは `max_abs_diff < 1e-3` を目標にする

### Phase 5: Security-oriented experiments

- prompt差分によるlogits変化を見る
- attentionの観察を可能にする
- 小規模なAI security実験の足場にする

---

## Related Docs

- [docs/transformer_notes.md](/Users/fumiyaoba/SakuraiLab/卒論/transformer_from_scratch/docs/transformer_notes.md:1)
- [docs/shape_table.md](/Users/fumiyaoba/SakuraiLab/卒論/transformer_from_scratch/docs/shape_table.md:1)
- [docs/experiment_report.md](/Users/fumiyaoba/SakuraiLab/卒論/transformer_from_scratch/docs/experiment_report.md:1)
- [docs/initial_issues.md](/Users/fumiyaoba/SakuraiLab/卒論/transformer_from_scratch/docs/initial_issues.md:1)

---

## License

ライセンスは未確定です。研究室方針と公開方針に合わせて後で明示します。
