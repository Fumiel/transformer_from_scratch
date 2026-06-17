---
marp: true
title: "Lesson 01: GitHub環境構築とTensor/Shapeの基礎"
theme: default
paginate: true
---

# Lesson 01

## GitHub環境構築とTensor/Shapeの基礎

Transformer from Scratch: Python Training + C++ Inference

---

# 今日のゴール

今日できるようになること：

1. repositoryをcloneしてテストを実行できる
2. Issue → branch → commit → PR の流れを理解する
3. scalar / vector / matrix / tensor を説明できる
4. Transformerで使うshape記号を説明できる
5. char-level tokenizerを実装・テストできる

---

# この10回で作るもの

```text
PythonでTiny Transformerを学習する
  ↓
学習済み重みをexportする
  ↓
C++で同じTransformerの推論を再現する
  ↓
Python/C++ logits差分を検証する
  ↓
AIセキュリティ実験に接続する
```

---

# なぜAIセキュリティでTransformerを実装するのか

LLMをAPIとして使うだけでは、次の問題を深く扱えない。

- prompt injection
- jailbreak
- activation monitoring
- logit analysis
- guardrail
- tool-use agent security

中身を触れるようになるために、まず小さいTransformerを自作する。

---

# GitHub運用

基本フロー：

```text
Issueを読む
  ↓
branchを切る
  ↓
実装する
  ↓
testを通す
  ↓
commitする
  ↓
PRを出す
  ↓
reviewを受ける
  ↓
mergeする
```

---

# Branch命名

```text
main             # 常に動く状態
dev              # 統合ブランチ
feat/tokenizer   # 機能追加
fix/test-ci      # 修正
docs/lesson01    # 資料追加
```

今日の例：

```bash
git checkout -b feat/tokenizer
```

---

# PRで必ず書くこと

```text
## Summary
何を実装したか

## Math / Shape
対応する数式とshape

## Test
どう確認したか

## Notes
詰まった点・設計判断
```

---

# Tensorの基礎

```text
scalar: x ∈ R
vector: x ∈ R^d
matrix: W ∈ R^{m×n}
tensor: X ∈ R^{B×T×C}
```

Transformerでは、ほぼ常にshapeを追う。

---

# 重要な記号

| 記号 | 意味 | 例 |
|---|---|---:|
| B | batch size | 4 |
| T | sequence length | 64 |
| C | embedding dimension | 128 |
| V | vocabulary size | 128 |
| H | number of heads | 4 |
| D | head dimension | 32 |

---

# Transformerで最もよく出るshape

```text
input_ids: [B, T]
token_emb: [B, T, C]
logits:    [B, T, V]
```

意味：

- B個のデータをまとめて処理する
- 各データはT個のtokenを持つ
- 各tokenをC次元ベクトルで表す
- 最後にV個のtoken候補へスコアを出す

---

# 例

```text
B = 2
T = 5
C = 4
```

```text
input_ids: [2, 5]

[
  [72, 101, 108, 108, 111],
  [87, 111, 114, 108, 100]
]
```

各tokenを4次元にすると：

```text
token_emb: [2, 5, 4]
```

---

# Tokenizerとは

モデルは文字列を直接読めない。

```text
"hello" → [104, 101, 108, 108, 111]
```

逆変換も必要。

```text
[104, 101, 108, 108, 111] → "hello"
```

---

# char-level tokenizer

文字を1つのtokenとして扱う。

```text
vocab = {"h":0, "e":1, "l":2, "o":3}

"hello" → [0, 1, 2, 2, 3]
```

利点：

- 実装が簡単
- tokenと文字の対応が見える
- 第1回にちょうどよい

---

# なぜ最初はchar-levelでよいか

本物のLLMではBPEやSentencePieceを使う。

でも、第1回で重要なのは以下：

```text
文字列
  ↓
token ids
  ↓
embedding
  ↓
Transformer
  ↓
logits
```

まずこの流れを理解する。

---

# 今日の実装対象

```text
python/tiny_transformer/tokenizer.py
python/tiny_transformer/config.py
tests/test_tokenizer.py
cpp/tensor.hpp
cpp/tensor.cpp
docs/shape_table.md
```

---

# 今日の完了条件

```text
python -m pytest が通る
C++がbuildできる
encode/decodeを説明できる
shape表をdocsに追記できる
PRを1つ出せる
```

---

# AIセキュリティへの接続

prompt injectionも、最初はtoken列です。

```text
"Ignore previous instructions..."
```

これはモデル内部では、

```text
[token_1, token_2, ..., token_T]
```

になり、embedding、attention、logitsへ流れる。

攻撃文字列がどのように出力分布を変えるかを見るために、まずtokenとshapeを理解する。

---

# 今日の問い

1. なぜ `input_ids` は `[B, T]` なのか？
2. なぜ `logits` は `[B, T, V]` なのか？
3. tokenizerの `encode` と `decode` は何をしているのか？
4. unknown character が来たとき、どう扱うべきか？
5. PythonとC++で同じTensorを扱うとき、何に気をつけるべきか？

---

# Hands-onへ

次は `exercises.md` に沿って作業します。
