# Lesson 01 宿題

## 目的

第2回のEmbedding/Linear実装に入る前に、以下を固めます。

- tokenizerを自分で説明できる
- Tensor shapeを追える
- GitHub PRの流れに慣れる
- C++ Tensorの土台を理解する

---

## Homework 1: tokenizerのテストを増やす

対象ファイル：

```text
tests/test_tokenizer.py
```

追加するテスト：

```text
1. roundtrip test
2. unknown character test
3. vocab_size test
4. empty string test
```

例：

```python
import pytest

from tiny_transformer.tokenizer import CharTokenizer


def test_unknown_character_raises_value_error():
    tok = CharTokenizer("abc")
    with pytest.raises(ValueError):
        tok.encode("abd")
```

完了条件：

```bash
python -m pytest
```

が通ること。

---

## Homework 2: docs/shape_table.mdを更新する

以下を追記してください。

```markdown
## Lesson 01: Basic Tensor Shapes

| Symbol | Meaning | Example |
|---|---|---:|
| B | batch size | 4 |
| T | sequence length | 64 |
| C | embedding dimension | 128 |
| V | vocabulary size | 128 |
| H | number of heads | 4 |
| D | head dimension | 32 |

| Tensor | Shape | Meaning |
|---|---|---|
| input_ids | [B, T] | token ids |
| token_emb | [B, T, C] | token embeddings |
| logits | [B, T, V] | next-token scores |
```

---

## Homework 3: C++ Tensor skeletonを改善する

対象ファイル：

```text
cpp/tensor.hpp
cpp/tensor.cpp
```

必要な機能：

```text
- shapeを保持する
- dataを保持する
- numel()を返す
- size(dim)を返す
```

想定インターフェース：

```cpp
class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;

    Tensor() = default;
    explicit Tensor(std::vector<int> shape);

    int numel() const;
    int size(int dim) const;
};
```

完了条件：

```bash
cmake -S . -B build
cmake --build build
./build/tiny_transformer_cpp
```

が通ること。

---

## Homework 4: 200字説明を書く

以下の問いに、各自200字程度で答えてください。

```text
Transformer実装でshapeを追うことがなぜ重要か？
```

保存先：

```text
docs/lessons/lesson01/submissions/<your_name>_reflection.md
```

例：

```markdown
# Lesson 01 Reflection: <your_name>

Transformerでは、入力token列がembeddingされ、attentionやMLPを通ってlogitsになるまで、各段階でshapeが変化する。shapeを誤ると、行列積やreshapeが壊れるだけでなく、意図しないtoken間の情報混合が起きる可能性がある。AIセキュリティでは、攻撃入力がどの経路で出力に影響するかを追う必要があるため、shape理解は基礎になる。
```

---

## PR条件

宿題は小さいPRで提出してください。

branch例：

```bash
git checkout -b feat/tokenizer-tests
```

PRタイトル例：

```text
test: add tokenizer edge case tests
```

PRに含めるもの：

```text
- 変更内容
- 対応する数式/shape
- 実行したテスト
- 詰まった点
```
