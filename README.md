# boatrace-predictor

競艇(ボートレース)の予想精度検証・バックテスト基盤。

## セットアップ

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## データ取得

[Boatrace Open API](https://github.com/BoatraceOpenAPI/api) の統合`api/v1`から、
出走表・直前情報(展示航走の実測風速/波高/展示タイム等)・結果をまとめて取得し、
`data/boatrace.db` (SQLite) に保存する。**2026-01-01以降のデータのみ対応**。

```powershell
.venv\Scripts\python.exe scripts\backfill.py --start 2026-01-01 --end 2026-07-27
```

対象会場は `src/boatrace_predictor/config.py` の `target_stadiums` で変更できる
(初期値: 24=大村, 3=江戸川)。全会場が毎日開催するわけではないので、
両会場が同時開催している日を確認してからバックフィルすると無駄がない。

## テスト

```powershell
.venv\Scripts\python.exe -m pytest
```

## 現状

データ収集基盤のみ実装済み。特徴量・予想モデル・バックテストはこれから。
詳細は [CLAUDE.md](CLAUDE.md) を参照。
