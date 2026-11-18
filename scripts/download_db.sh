#!/usr/bin/env bash
# リリース db-latest からDBを取得する。壊れていたり無ければ、差し替え途中の新版
# (boatrace-new.db)、直前の版(boatrace-prev.db)の順に代わりを使う。各3回まで再試行。
# 使い方: bash scripts/download_db.sh <出力先> <owner/repo>
set -uo pipefail
out="$1"; repo="$2"; mkdir -p "$(dirname "$out")"
for name in boatrace.db boatrace-new.db boatrace-prev.db; do
  for i in 1 2 3; do
    rm -f "$out"
    if gh release download db-latest --repo "$repo" --pattern "$name" --output "$out" --clobber; then
      if python3 -c "import os,sqlite3,sys; f=sys.argv[1]; sys.exit(0 if os.path.getsize(f)>100_000_000 and sqlite3.connect(f).execute('pragma quick_check').fetchone()[0]=='ok' else 1)" "$out"; then
        echo "使用したDB: $name"; exit 0
      fi
      echo "$name は壊れている(再試行)"
    fi
    sleep 5
  done
done
echo "使えるDBが見つからない"; exit 1
