# Transformer from Scratch

Pythonで小さなDecoder-only Transformerを学習し、その重みを使ってC++で同等の推論器を再現する個人実装プロジェクトです。
目的は、Transformerの内部構造を実装レベルで理解しつつ、将来的にAIエージェントセキュリティ実験へ接続できる最小基盤を自分の手で作ることです。

現時点では、Python側の基本部品と最小モデル構造を先に固め、C++側はforward再現のための土台を整えている段階です。

## Table of Contents

- [Project Overview](#project-overview)
- [Why This Project](#why-this-project)
- [Current Scope](#current-scope)
- [What Is Custom vs PyTorch](#what-is-custom-vs-pytorch)
- [Trainable Parameters](#trainable-parameters)
- [Relation To Modern GPT](#relation-to-modern-gpt)
- [Current Status](#current-status)
- [Development Policy](#development-policy)
- [Shape Table](#shape-table)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Run tests](#run-tests)
  - [Python training](#python-training)
  - [Text generation](#text-generation)
  - [Internal representation tracing](#internal-representation-tracing)
  - [Train + generation log](#train--generation-log)
  - [Export weights](#export-weights)
  - [Compare Python and C++ logits](#compare-python-and-c-logits)
  - [C++ smoke test](#c-smoke-test)
- [Planned Milestones](#planned-milestones)
- [Related Docs](#related-docs)
- [License](#license)

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
- `train.py` は最小学習ループ、checkpoint保存、固定probeの定期trace保存に対応済み
- `generate.py` はcheckpoint読込、greedy decoding、生成ステップごとのtrace取得・保存に対応済み
- `tracing.py` はembedding、block、attention、FFN、logits、生成メタデータのtrace用dataclassを提供
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
- traceのshape、causal mask、residual加算、top-k、生成ステップ、固定probe保存を検証するpytestを追加済み
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
| `pos_emb` | `[1, T, C]` | batch方向にbroadcastするposition embedding |
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
  │   ├─ run_experiment.py
  │   ├─ export_weights.py
  │   ├─ compare_cpp.py
  │   └─ tiny_transformer/
  │       ├─ tokenizer.py
  │       ├─ config.py
  │       ├─ layers.py
  │       ├─ attention.py
  │       ├─ model.py
  │       └─ tracing.py
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
  │   ├─ test_tracing.py
  │   ├─ test_run_experiment.py
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
python python/train.py \
  --data data/tiny_corpus.txt \
  --steps 1000 \
  --checkpoint checkpoints/tiny.pt
```

学習中の内部表現を定点観測する場合は、`--trace-dir` を指定します。
`--probe-text` は学習データではなく、optimizer更新後のモデルへ定期的に入力する観測専用テキストです。
未指定時は学習コーパス先頭の最大32 tokenを固定probeとして使います。

```bash
python python/train.py \
  --data data/tiny_corpus.txt \
  --steps 1000 \
  --checkpoint checkpoints/tiny.pt \
  --trace-dir outputs/traces/train_001 \
  --probe-text "hello" \
  --trace-top-k 5
```

training batchは指定した `--batch-size` のままです。trace用forwardだけを `B=1`、
`model.eval()`、`torch.no_grad()` で実行し、学習のloss・勾配・optimizer更新には使いません。
`--trace-dir` が空でない場合は、異なる実験のtraceが混在しないようにエラーになります。
既定の保存stepは、初期変化と学習後半を比較できるように、step 1、step 10、
全stepの10分割位置、最終stepです。`--steps 1000` なら
`1, 10, 100, 200, ..., 1000` で保存します。

```text
outputs/traces/train_001/
├─ metadata.json
├─ metrics.jsonl
├─ step_000001_probe.pt
├─ step_000010_probe.pt
└─ step_001000_probe.pt
```

### Text generation

`generate.py` は `train.py` が保存したcheckpointを読み込み、promptからgreedy decodingで文字を生成します。
系列が `block_size` を超える場合は、末尾 `block_size` tokenだけを文脈としてモデルへ渡します。

```bash
python python/generate.py \
  --checkpoint checkpoints/tiny.pt \
  --prompt "hello" \
  --max-new-tokens 100
```

生成ステップごとのtraceを保存する場合は `--trace-output` を指定します。

```bash
python python/generate.py \
  --checkpoint checkpoints/tiny.pt \
  --prompt "hello" \
  --max-new-tokens 20 \
  --trace-output outputs/traces/inference_hello.pt \
  --trace-top-k 5
```

### Internal representation tracing

通常の学習・生成コマンドは従来どおりです。内部表現は、CLIでtrace保存先を指定するか、
Python APIで `return_trace=True` を明示した場合だけ取得します。通常の
`model(input_ids)` や `generate_text(...)` は、従来どおり logits または生成文字列だけを返します。

`model(input_ids, return_trace=True)` で取得する `ModelTrace` には次の情報が含まれます。

| 区分 | 主な取得内容 | shape |
|---|---|---|
| 入力 | token ID、token embedding、position embedding、その和 | `[B,T]`, `[B,T,C]` |
| block | block入力・出力、LayerNorm出力 | `[B,T,C]` |
| attention | Q/K/V、score、mask後score、probability、head出力、結合後出力、出力射影 | `[B,H,T,D]`, `[B,H,T,T]`, `[B,T,C]` |
| FFN | `fc1` 出力、GELU出力、`fc2` 出力 | `[B,T,4C]`, `[B,T,C]` |
| residual | attention/FFNの加算量と加算後の表現 | `[B,T,C]` |
| 出力 | final LayerNorm出力、全logits、各位置のtop-k ID/logit/確率 | `[B,T,C]`, `[B,T,V]`, `[B,T,K]` |

trace内のtensorは `detach()`、CPU転送、`clone()` を行った独立したsnapshotです。
そのため計算グラフやGPU/MPSメモリを保持しません。これは推論時の観測用であり、
activationに対する勾配解析には使いません。

checkpointから1回のforwardの内部表現を取得する例です。

```bash
PYTHONPATH=python python - <<'PY'
from pathlib import Path
import torch

from generate import load_checkpoint

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model, tokenizer, _ = load_checkpoint(Path("checkpoints/tiny.pt"), device)
input_ids = torch.tensor([tokenizer.encode("hello")], dtype=torch.long, device=device)

logits, trace = model(input_ids, return_trace=True, top_k=5)
print(trace.token_embedding.shape)                 # [1, T, C]
print(trace.blocks[0].attention.probabilities.shape)  # [1, H, T, T]
print(trace.top_k_ids[0, -1].tolist())

Path("outputs").mkdir(exist_ok=True)
torch.save(trace, "outputs/forward_trace.pt")
PY
```

生成中の各ステップを観測する場合は、`generate_text` に `return_trace=True` を渡します。
`GenerationTrace.steps` の各要素には、切り詰め後のcontext、元の生成列における
`context_start`、選択token、最終位置のtop-k、対応する `ModelTrace` が含まれます。

```bash
PYTHONPATH=python python - <<'PY'
from pathlib import Path
import torch

from generate import generate_text, load_checkpoint

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model, tokenizer, config = load_checkpoint(Path("checkpoints/tiny.pt"), device)
generated, trace = generate_text(
    model=model,
    tokenizer=tokenizer,
    config=config,
    prompt="hello",
    max_new_tokens=20,
    device=device,
    return_trace=True,
    trace_top_k=5,
)
print(generated)
print(trace.steps[0].top_k)

Path("outputs").mkdir(exist_ok=True)
torch.save(trace, "outputs/generation_trace.pt")
PY
```

`top_k` / `trace_top_k` は、次token候補のうち logit が高い順に残す個数です。
語彙数より大きい値を指定した場合は、語彙数に制限されます。traceは入力token列や
内部表現を含むため、機密データを使う実験では保存先と共有範囲を管理してください。

保存したtraceはdataclassを含むため、信頼できる自分の実験ファイルに限り、次のように読み込みます。

```python
# generate.py が保存した GenerationTrace
generation_trace = torch.load(
    "outputs/traces/inference_hello.pt",
    map_location="cpu",
    weights_only=False,
)

# train.py が保存したstep情報・loss・ModelTrace
probe_payload = torch.load(
    "outputs/traces/train_001/step_000100_probe.pt",
    map_location="cpu",
    weights_only=False,
)
probe_trace = probe_payload["trace"]
```

`run_experiment.py` は現在、学習設定・最終loss・生成文を
`outputs/generation_log.md` に記録しますが、trace関連のCLI引数は中継しません。
学習のみ・推論のみのtrace保存には、上記の `train.py` / `generate.py` を使います。

### Train + generation log

`run_experiment.py` は学習、生成、結果ログ保存を1回で実行します。
checkpointごとの比較を残したいときは、`--checkpoint` と `--output-log` を指定して使います。

```bash
python python/run_experiment.py \
  --data data/tiny_corpus.txt \
  --steps 200 \
  --batch-size 4 \
  --n-layer 1 \
  --n-head 2 \
  --n-embd 8 \
  --checkpoint checkpoints/tiny_200.pt \
  --prompt "hello" \
  --max-new-tokens 40 \
  --output-log outputs/generation_log.md
```

このスクリプトは次を行います。

- `train.py` 相当の学習を実行してcheckpointを保存する
- 保存したcheckpointを読み込んで `generate.py` 相当のgreedy decodingを行う
- 実行条件と生成結果を `outputs/generation_log.md` に追記する

checkpoint比較の基本手順:

1. `--steps` と `--checkpoint` を変えて複数回実行する
2. prompt は同じにそろえる
3. `outputs/generation_log.md` の生成結果と `final_loss` を見比べる

例:

```bash
python python/run_experiment.py --steps 200 --checkpoint checkpoints/tiny_200.pt --prompt "hello"
python python/run_experiment.py --steps 500 --checkpoint checkpoints/tiny_500.pt --prompt "hello"
python python/run_experiment.py --steps 1000 --checkpoint checkpoints/tiny_1000.pt --prompt "hello"
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
