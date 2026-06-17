# Lesson 01 完了チェックリスト

## 受講者チェック

### 環境構築

- [ ] repositoryをcloneできた
- [ ] Python virtual environmentを作成できた
- [ ] `pip install -e .[dev]` を実行できた
- [ ] `python -m pytest` が通った
- [ ] `cmake -S . -B build` が通った
- [ ] `cmake --build build` が通った
- [ ] C++ executableを実行できた

### GitHub運用

- [ ] Issueを読んだ
- [ ] branchを切った
- [ ] commitを作った
- [ ] pushした
- [ ] PRを作った
- [ ] PR本文にSummary/Test/Notesを書いた

### 数式・shape理解

- [ ] scalar / vector / matrix / tensor の違いを説明できる
- [ ] B, T, C, V, H, D の意味を説明できる
- [ ] `input_ids: [B, T]` を説明できる
- [ ] `token_emb: [B, T, C]` を説明できる
- [ ] `logits: [B, T, V]` を説明できる
- [ ] `C = H × D` を説明できる

### tokenizer理解

- [ ] `encode` の役割を説明できる
- [ ] `decode` の役割を説明できる
- [ ] `stoi` と `itos` を説明できる
- [ ] `decode(encode(text)) = text` の意味を説明できる
- [ ] unknown character の扱いを説明できる

### C++ Tensor理解

- [ ] Tensorがshapeとdataを持つことを説明できる
- [ ] 多次元配列が1次元メモリに並ぶことを説明できる
- [ ] row-major layoutを説明できる
- [ ] `numel()` の意味を説明できる

---

## レビュー観点

PR reviewerは以下を確認してください。

### Code

- [ ] 小さい変更になっているか
- [ ] 関数名・変数名が分かりやすいか
- [ ] 不要なファイルがcommitされていないか
- [ ] `.venv/`, `build/`, `__pycache__/` が含まれていないか

### Test

- [ ] `python -m pytest` が通っているか
- [ ] C++ buildが通っているか
- [ ] 追加した機能に対応するテストがあるか

### Docs

- [ ] shapeが明記されているか
- [ ] 数式または対応する説明があるか
- [ ] PR本文に実行コマンドが書かれているか

---

## 第1回のDefinition of Done

第1回終了時点で、repositoryに以下があること。

```text
- docs/lessons/lesson01/ の講義資料
- docs/shape_table.md の基本shape表
- tokenizerの基本テスト
- C++ Tensor skeleton
- 少なくとも1つのPR履歴
```

受講者が以下を口頭で説明できること。

```text
1. なぜTransformerではshapeが重要なのか
2. tokenizerは何をしているのか
3. input_ids, token_emb, logitsのshape
4. PythonとC++で同じモデルを再現する意義
5. AIセキュリティにtoken/shape理解がどうつながるか
```
