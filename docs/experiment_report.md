# Experiment Report

## 目的

Pythonで学習したTiny TransformerのforwardをC++推論器で再現し、logitsの差分を測る。

## 実験設定

| 項目 | 値 |
|---|---|
| dataset | `data/tiny_corpus.txt` |
| tokenizer | char-level |
| block size | 64 |
| n_layer | 2 |
| n_head | 2 |
| n_embd | 64 |

## 結果

| 実装 | loss | max_abs_diff | 備考 |
|---|---:|---:|---|
| Python baseline | TBD | - | - |
| C++ inference | - | TBD | - |

## 考察

TBD

## AIセキュリティへの接続

TBD
