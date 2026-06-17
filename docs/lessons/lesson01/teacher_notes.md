# Lesson 01 教員・TA用進行メモ

## この回で重視すること

第1回は、難しい数式を詰め込む回ではありません。目的は、後輩が今後の実装で迷子にならないように、以下の基礎姿勢を作ることです。

```text
shapeを必ず書く
小さいPRで出す
testを通してからmergeする
分からないことをdocsに残す
```

特に、Transformer実装ではshapeを追えないと必ず詰まります。第1回では、実装量よりも「shapeを言葉で説明できること」を重視してください。

---

## 最初に見せる完成イメージ

冒頭で以下の流れを見せると、全体像が伝わりやすいです。

```bash
python python/train.py --data data/tiny_corpus.txt
python python/generate.py --prompt "hello"
python python/export_weights.py --checkpoint checkpoints/tiny.pt --out weights/tiny_weights.json
./build/tiny_transformer_cpp --weights weights/tiny_weights.json --prompt "hello"
python python/compare_cpp.py --checkpoint checkpoints/tiny.pt --cpp-output outputs/cpp_logits.txt
```

第1回時点では全部動かなくても構いません。最終到達点として見せます。

---

## 説明のコツ

### Tensor

「Tensorとは多次元配列」と言うだけで終わらせず、必ずshapeを具体例で書かせます。

```text
B=2, T=5, C=4 のとき token_emb は [2,5,4]
```

### Tokenizer

本物のLLMのtokenizerに深入りしないでください。BPEやSentencePieceは名前だけでよいです。

第1回では、

```text
文字列 → 整数列 → embeddingへ渡す
```

が分かれば十分です。

### C++ Tensor

最初から高機能Tensor classにしないでください。目的は、PyTorchが隠しているshape/dataの概念をC++で見えるようにすることです。

---

## よくある詰まり

### 1. import error

症状：

```text
ModuleNotFoundError: No module named 'tiny_transformer'
```

対応：

```bash
pip install -e .[dev]
```

または、virtual environmentが有効か確認します。

### 2. pytestが見つからない

対応：

```bash
pip install -e .[dev]
```

### 3. CMakeがない

macOS:

```bash
brew install cmake
```

Ubuntu/WSL:

```bash
sudo apt update
sudo apt install cmake build-essential
```

Windowsの場合は、Visual Studio Build Tools または WSL を推奨します。

### 4. build/をcommitしそうになる

`.gitignore` に `build/` があることを確認させてください。

```bash
git status
```

を必ず見る習慣をつけます。

---

## その場で出す質問

理解確認として、以下を聞いてください。

```text
1. input_idsが[B,T]で、[T,B]ではない理由は？
2. logitsの最後の次元がCではなくVなのはなぜ？
3. decode(encode(text)) = text が壊れると何が困る？
4. C++でshapeを自分で持つ必要があるのはなぜ？
5. prompt injectionの文字列はモデル内部で何になる？
```

---

## 採点の目安

| 観点 | 点 |
|---|---:|
| 環境構築できた | 20 |
| GitHub PRを出せた | 20 |
| Tensor/shapeを説明できた | 25 |
| tokenizerを説明できた | 20 |
| AIセキュリティ接続を説明できた | 15 |

合計100点。

第1回はコード量よりも、開発フローとshape理解を評価してください。
