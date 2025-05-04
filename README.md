# 動学最適化シミュレーションプロジェクト

このリポジトリは、ベルマン方程式を用いた動学最適化モデルのシミュレーションコードと分析を含んでいます。

## 環境設定

このプロジェクトはuv、pytask、pytestを使用して環境・タスク・テストを管理しています。
セットアップ方法については、docs/setup.md を参照してください。

## プロジェクト構造

- `src/`: ソースコード
  - `model/`: モデル定義
  - `simulation/`: シミュレーション実行用スクリプト
  - `utils/`: ユーティリティ関数
- `tests/`: テストコード
- `notebooks/`: 探索的分析用Jupyter notebooks
- `docs/`: ドキュメント
- `results/`: シミュレーション結果・図表

## 実行方法

```bash
# 環境のセットアップ
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip sync requirements.txt

# 全タスクの実行
pytask

# テストの実行
pytest
```
