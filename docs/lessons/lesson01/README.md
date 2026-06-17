# Lesson 01: GitHub環境構築とTensor/Shapeの基礎

## 目的

第1回では、Transformer実装に入る前の基礎として、開発環境、GitHub運用、Tensorとshape、char-level tokenizerを扱います。

この回のゴールは、後輩が以下を説明・実行できるようになることです。

```text
1. repositoryをcloneしてPython/C++のテストを実行できる
2. GitHub Issue → branch → commit → PR の流れを理解する
3. scalar / vector / matrix / tensor の違いを説明できる
4. Transformerで使う主要なshape記号 B, T, C, V, H, D を説明できる
5. 文字列をtoken idsに変換し、元の文字列に戻せる
```

## 格納ファイル

```text
docs/lessons/lesson01/
  README.md          # このファイル。第1回の概要
  slides.md          # 講義用スライド原稿
  lecture_notes.md   # 詳細講義ノート
  exercises.md       # 当日のハンズオン
  homework.md        # 次回までの宿題
  checklist.md       # 完了条件・レビュー観点
```

## 推奨時間配分

| 時間 | 内容 |
|---:|---|
| 0:00-0:15 | 目的共有・完成物デモ |
| 0:15-0:35 | GitHub運用説明 |
| 0:35-1:05 | Tensorとshapeの数式理解 |
| 1:05-1:25 | char-level tokenizerの数式と実装 |
| 1:25-1:55 | ハンズオン |
| 1:55-2:15 | テスト・PR作成 |
| 2:15-2:30 | 振り返り・宿題確認 |

## 対応Issue

- #1 repo初期化: Python/C++/CIの土台を作る
- #2 char-level tokenizerを実装する
- #10 C++: Tensorクラスのskeletonを実装する
- docs: lesson01のshape表と学習メモを追加する

## 当日の成果物

- `python -m pytest` が通る
- `cmake -S . -B build && cmake --build build` が通る
- `CharTokenizer.encode()` / `CharTokenizer.decode()` の動作を説明できる
- `docs/shape_table.md` に第1回のshape表が追記される
- 各自が小さいPRを1つ出す
