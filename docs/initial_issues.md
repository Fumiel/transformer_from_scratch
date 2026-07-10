# Initial Issues

## Milestone 0: Repository setup

### #1 repo初期化: Python/C++/CIの土台を作る

目的: メンバーが clone してすぐ作業できる状態にする。  
完了条件:

- `pytest` が通る
- `cmake --build build` が通る
- GitHub Actions が通る

---

## Milestone 1: Python components

### #2 char-level tokenizerを実装する

目的: 最小のtokenizerを理解する。  
完了条件:

- `encode -> decode` の roundtrip test が通る
- 未知文字の扱いを説明できる

### #3 Python: Linear, LayerNorm, GELUを実装する

目的: Transformerの基本layerを自作する。  
完了条件:

- shapeコメントがある
- PyTorch referenceとの数値比較testを書く

### #4 Python: scaled dot-product attentionを実装する

目的: attentionの中核を理解する。  
完了条件:

- `q @ k^T / sqrt(d)` を実装
- causal maskを実装
- 出力shapeをtestする

### #5 Python: multi-head causal attentionを実装する

目的: head分割と結合を理解する。  
完了条件:

- `[B,T,C] -> [B,H,T,D]` の変換を説明できる
- attention出力が `[B,T,C]` になる

### #6 Python: TransformerBlockを実装する

目的: LayerNorm、Attention、FFN、Residualを統合する。  
完了条件:

- 入出力shapeが `[B,T,C]`
- block単体testが通る

### #7 Python: training loopを実装する

目的: next-token predictionの学習を理解する。  
完了条件:

- lossが下がる
- checkpoint保存ができる

### #8 Python: generate.pyを実装する

目的: logitsからtokenを生成する流れを理解する。  
完了条件:

- greedy decodingができる
- top-k samplingは発展課題

実装状況:

- checkpoint読込、config / vocab復元、`TinyGPT` への重みロードは実装済み
- greedy decoding、生成tokenの連結、decode、`block_size` 超過時の文脈切り詰めは実装済み

### #9 Python: 重みexportを実装する

目的: PyTorch checkpointをC++で読める形にする。  
完了条件:

- tensor名、shape、flat dataを保存できる
- small tensorで読み戻しtestを書く

---

## Milestone 2: C++ inference

### #10 C++: Tensorクラスを実装する

目的: shapeとrow-major配列を理解する。  
完了条件:

- `numel()` が正しい
- shape mismatch時にassertできる

### #11 C++: Linearを実装する

目的: matrix multiplicationを理解する。  
完了条件:

- Python Linearと出力差分を比較する

### #12 C++: LayerNormを実装する

目的: mean/variance/epsの数値処理を理解する。  
完了条件:

- Python LayerNormと出力差分を比較する

### #13 C++: Softmax + causal maskを実装する

目的: 数値安定softmaxを理解する。  
完了条件:

- max subtractionを入れる
- mask後の未来token確率が0になる

### #14 C++: Attentionを実装する

目的: C++でcausal self-attentionを再現する。  
完了条件:

- shapeをコメントで説明する
- small inputでPythonと比較する

### #15 C++: TransformerBlockを実装する

目的: full blockをC++で統合する。  
完了条件:

- Python blockと出力差分を比較する

### #16 C++: Python重みを読み込む

目的: exportされた重みを推論器に接続する。  
完了条件:

- tensor名でweightを取得できる
- shape validationできる

### #17 Python/C++のforward一致テストを作る

目的: 実装の正しさを数値で保証する。  
完了条件:

- 同じinput tokensでlogitsを比較
- `max_abs_diff < 1e-3` を目標にする

### #18 READMEと実験レポートを書く

目的: 就活・研究室内共有に使える成果物にする。  
完了条件:

- 実行方法がREADMEにある
- 実験条件と結果がdocsにある
- demoログが貼られている
