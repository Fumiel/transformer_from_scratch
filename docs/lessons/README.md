# Lessons

このディレクトリには、全10回の実装レッスン資料を格納します。

各回は次の構成で管理します。

```text
lessonXX/
  README.md          # その回の概要・到達目標・時間配分
  slides.md          # 講義用スライド原稿。Marp等に変換可能
  lecture_notes.md   # 受講者向けの詳細講義ノート
  exercises.md       # 当日のハンズオン課題
  homework.md        # 次回までの宿題・PR課題
  checklist.md       # 完了条件・レビュー観点
```

## 全体方針

毎回、次の流れで進めます。

```text
数式理解 → shape確認 → Python実装 → C++実装/検証 → test → PR → docs更新
```

実装だけで終わらせず、各PRには以下を含めます。

- 対応する数式
- 入力shape / 出力shape
- 実行したテスト
- 詰まった点
- AIセキュリティへの接続
