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

## What Is Custom vs PyTorch

この実装では、Transformerの計算フローそのものは自作しつつ、テンソル計算基盤と一部の基本部品にPyTorchを使っています。

### Forward Path Summary

| step | role | custom implementation | PyTorch usage |
|---|---|---|---|
| Input handling | `input_ids` を受け取り系列長を検証 | `TinyGPT.forward` の制御フロー、`block_size` 超過時の例外処理 | `torch.Tensor` の shape 操作、`torch.arange` |
| Token / Position Embedding | token埋め込みと位置埋め込みを作って加算 | forward内での接続方法 | `nn.Embedding` |
| Transformer block repetition | blockを `n_layer` 回適用 | blockを順に適用するループ | `nn.ModuleList` |
| LayerNorm | 最後の次元で平均・分散を計算し正規化、affine変換 | `LayerNorm` を自作 | `nn.Parameter`, `torch.mean`, `torch.var`, `torch.sqrt` |
| Linear projection | 線形変換 `x @ W^T + b` | `Linear` を自作 | `nn.Parameter`, `torch.empty`, `torch.nn.init.kaiming_uniform_` |
| Attention mask | 因果マスクを作成 | `causal_mask` を自作 | `torch.tril`, `torch.ones`, `view` |
| Scaled dot-product attention | score計算、スケーリング、mask適用、softmax、value集約 | `scaled_dot_product_attention` を自作 | 行列積 `@`, `transpose`, `masked_fill`, `torch.softmax` |
| Multi-head attention | QKV作成、head分割、attention計算、head結合、出力射影 | `MultiHeadCausalSelfAttention` を自作 | `split`, `view`, `transpose`, `contiguous` |
| Feed Forward | `fc1 -> GELU -> fc2` | `FeedForward` を自作 | 内部で自作 `Linear` とPyTorchテンソル演算を使用 |
| GELU | GPT系でよく使うtanh近似 | `gelu` を自作 | `torch.tanh` |
| Residual connections | attention後・ffn後の残差加算 | `TransformerBlock.forward` を自作 | テンソル加算 |
| Final normalization and LM head | 最終LayerNormと語彙方向への射影 | forwardの接続、自作 `LayerNorm` / `Linear` | PyTorchテンソル演算 |

### High-Level Policy

- 高レベルAPIの `torch.nn.Transformer` は使っていません。
- 高レベルAPIの `torch.nn.MultiheadAttention` は使っていません。
- Transformerの内部ロジックは、shapeを追いやすい形で自分で組んでいます。
- PyTorchは主に、テンソル演算、パラメータ管理、埋め込み層、初期化に使っています。

## Trainable Parameters

`train.py` では `optimizer = torch.optim.AdamW(model.parameters(), ...)` を使っているため、
学習対象は `TinyGPT` の `nn.Parameter` 全体です。クラス構造に沿うと、主に次の重みが更新されます。

```text
TinyGPT
├─ token_embedding.weight
├─ position_embedding.weight
├─ blocks[i].ln1.weight
├─ blocks[i].ln1.bias
├─ blocks[i].attn.qkv_proj.weight
├─ blocks[i].attn.qkv_proj.bias
├─ blocks[i].attn.out_proj.weight
├─ blocks[i].attn.out_proj.bias
├─ blocks[i].ln2.weight
├─ blocks[i].ln2.bias
├─ blocks[i].ffn.fc1.weight
├─ blocks[i].ffn.fc1.bias
├─ blocks[i].ffn.fc2.weight
├─ blocks[i].ffn.fc2.bias
├─ final_ln.weight
├─ final_ln.bias
└─ lm_head.weight
```

補足:

- `blocks[i]` は各 Transformer block を表します。`n_layer=2` なら `blocks[0]` と `blocks[1]` があります。
- 学習対象ではないものは `input_ids`, `x`, `y`, `logits`, `loss`, `block_size`, tokenizer の `stoi/itos`, causal mask などです。
- `lm_head` は `bias=False` なので、ここでは `weight` だけが学習されます。

## Relation To Modern GPT

この実装は教育用の最小構成ですが、現代のGPT系モデルと共通する骨格をいくつか持っています。一方で、実運用で重要になる高速化や安定化の工夫はかなり省いています。

### Shared Design Choices

- `Decoder-only Transformer` 構成です。
- 次トークン予測のための `causal self-attention` を使っています。
- `token embedding + position embedding` で入力表現を作っています。
- `LayerNorm -> Attention -> Residual -> LayerNorm -> FeedForward -> Residual` というblock構造を採用しています。
- FeedForwardは `n_embd -> 4 * n_embd -> n_embd` の拡張・圧縮構成です。
- 最終 `LayerNorm` の後に `LM head` で語彙方向へ射影してlogitsを出します。

### Simplifications Compared To Modern GPT

| area | this repository | modern GPT-style models |
|---|---|---|
| Positional encoding | 学習可能な絶対位置埋め込み | RoPE などの回転位置埋め込みが主流 |
| FeedForward activation | GELU | SwiGLU / GeGLU などがよく使われる |
| Attention implementation | 素朴なscaled dot-product attention | FlashAttention系など高速化実装が一般的 |
| Inference optimization | KV cacheなし | 長い生成ではKV cacheがほぼ必須 |
| Regularization | dropoutなし | 学習時は各種正則化や安定化が入ることが多い |
| Scale-oriented tweaks | 最小限 | 重み共有、初期化調整、normの細部最適化などが入ることが多い |

このため、本実装は「現代GPTの本質的な計算の流れを学ぶための最小モデル」であり、「現代の大規模GPTの完全な再現」ではありません。

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
- `train.py` は最小学習ループとcheckpoint保存まで実装済み
- `generate.py` はcheckpoint読込とgreedy decodingまで実装済み
- `export_weights.py` は未実装
- `compare_cpp.py` は未実装

### C++

- `Tensor` / layer / modelの骨組みは配置済み
- `main.cpp` にはsmoke test相当の入口がある
- `TinyGPTCpp::forward` はまだゼロlogitsを返す仮実装
- attention / softmax / weights loader は未実装

### Tests

- tokenizerのpytestは存在
- `Linear` / `LayerNorm` / `GELU` のpytestを追加済み
- causal mask / attention / `FeedForward` / `TransformerBlock` / `TinyGPT` のpytestを追加済み
- `train.py` のbatch生成、checkpoint保存、短い学習実行のpytestを追加済み
- `generate.py` のcheckpoint復元、greedy decoding、文脈切り詰めのpytestを追加済み
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
  │   ├─ test_attention.py
  │   ├─ test_layers.py
  │   ├─ test_tokenizer.py
  │   ├─ test_feedforward.py
  │   ├─ test_transformer_block.py
  │   ├─ test_generate.py
  │   ├─ test_tiny_gpt.py
  │   └─ test_train.py
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

`train.py` は最小の next-token prediction 学習ループを持ちます。
テキスト読込、tokenize、ランダム subsequence batch 作成、loss 計算、optimizer 更新、checkpoint 保存まで実装済みです。

```bash
python python/train.py --data data/tiny_corpus.txt --steps 1000
```

### Text generation

`generate.py` は `train.py` が保存したcheckpointを読み込み、promptからgreedy decodingで文字を生成します。
系列が `block_size` を超える場合は、末尾 `block_size` tokenだけを文脈としてモデルへ渡します。

```bash
python python/generate.py --checkpoint checkpoints/tiny.pt --prompt "hello" --max-new-tokens 100
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

- training loopを実装済み
- lossが下がる最小学習条件を作成済み
- 文字レベル生成をgreedy decodingで実装済み

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
