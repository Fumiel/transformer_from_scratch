# Lesson 01 ハンズオン課題

## 事前準備

repositoryをcloneします。

```bash
git clone git@github.com:<org-or-user>/transformer-from-scratch.git
cd transformer-from-scratch
```

Python環境を作ります。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
pip install torch
```

C++をbuildします。

```bash
cmake -S . -B build
cmake --build build
```

---

## Exercise 1: repositoryの状態確認

以下を実行してください。

```bash
python -m pytest
```

期待：

```text
tests/test_tokenizer.py が通る
```

次にC++を実行します。

```bash
./build/tiny_transformer_cpp
```

期待：

```text
C++ executableが起動する
```

### PRに書くこと

```text
- 実行したコマンド
- 結果
- エラーが出た場合は原因と対応
```

---

## Exercise 2: tokenizerのroundtripを確認する

Python REPLまたは小さいscriptで確認してください。

```python
from tiny_transformer.tokenizer import CharTokenizer

tok = CharTokenizer("hello world")
ids = tok.encode("hello")
text = tok.decode(ids)

print(ids)
print(text)
print(tok.vocab_size)
```

### 考えること

1. `ids` はなぜ整数列になるか
2. `vocab_size` は何を表すか
3. `decode(encode("hello")) == "hello"` がなぜ重要か

---

## Exercise 3: unknown characterの挙動を見る

以下を実行してください。

```python
from tiny_transformer.tokenizer import CharTokenizer

tok = CharTokenizer("abc")
print(tok.encode("abd"))
```

期待：

```text
ValueError: unknown character: 'd'
```

### 考えること

1. なぜ未知文字を無視しないのか
2. 実サービスなら未知文字をどう扱う選択肢があるか
3. AIセキュリティ上、未知文字や特殊文字はなぜ重要か

---

## Exercise 4: shapeを書き出す

次の設定を考えます。

```text
B = 4
T = 16
C = 64
V = 128
```

以下のshapeを書いてください。

```text
input_ids: ?
token_emb: ?
logits: ?
```

答え：

```text
input_ids: [4, 16]
token_emb: [4, 16, 64]
logits:    [4, 16, 128]
```

次に、以下を `docs/shape_table.md` に追記してください。

```markdown
## Lesson 01: Basic shapes

| Name | Shape | Meaning |
|---|---|---|
| input_ids | [B, T] | token id sequence |
| token_emb | [B, T, C] | token embeddings |
| logits | [B, T, V] | next-token scores |
```

---

## Exercise 5: configの制約を確認する

以下を実行してください。

```python
from tiny_transformer.config import TinyGPTConfig

cfg = TinyGPTConfig(n_embd=64, n_head=2)
print(cfg)

bad = TinyGPTConfig(n_embd=63, n_head=2)
```

期待：

```text
n_embd must be divisible by n_head
```

### 考えること

Multi-head attentionでは、

```text
C = H × D
```

です。

`n_embd=63, n_head=2` がダメな理由を説明してください。

---

## Exercise 6: C++ Tensor skeletonを読む

以下のファイルを読んでください。

```text
cpp/tensor.hpp
cpp/tensor.cpp
```

確認すること：

```text
- Tensorは何をメンバ変数として持っているか
- shapeはどの型で持っているか
- dataはどの型で持っているか
- numel() は何を返すべきか
```

### 追加課題

もし未実装なら、`numel()` を実装してください。

仕様：

```text
shape = [2, 3, 4] のとき numel() = 24
shape = [5] のとき numel() = 5
```

---

## Exercise 7: branchを切ってPRを作る

作業branchを作ります。

```bash
git checkout -b docs/lesson01-shape-table
```

変更します。

```bash
git add docs/shape_table.md
git commit -m "docs: add lesson 01 shape table"
git push -u origin docs/lesson01-shape-table
```

GitHub上でPRを作成してください。

### PR本文テンプレート

```markdown
## Summary
Lesson 01で扱った基本shapeをdocsに追記しました。

## Math / Shape
- input_ids: [B, T]
- token_emb: [B, T, C]
- logits: [B, T, V]

## Test
- python -m pytest
- cmake -S . -B build && cmake --build build

## Notes
Transformer実装ではshapeを追うことが重要だと理解しました。
```

---

## 追加発展課題

余裕がある人は以下を実装してください。

### A. tokenizerのテストを追加

`tests/test_tokenizer.py` に以下の観点を追加します。

```text
- unknown characterでValueErrorになる
- vocab_sizeが正しい
- 空文字列をencode/decodeできる
```

### B. C++ Tensorのテスト方針を書く

まだC++ test frameworkを入れていない場合、`docs/lessons/lesson01/homework.md` にC++ Tensorをどうテストするかを書いてください。

観点：

```text
- shapeが保存される
- dataの要素数がshapeと一致する
- numel()が正しい
- 範囲外アクセスをどう扱うか
```
