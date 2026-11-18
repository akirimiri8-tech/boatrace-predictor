#!/usr/bin/env bash
# DBをリリース db-latest に安全にアップロードする。
# 2026-11-18: `gh release upload --clobber` は「古いアセットを消してから上げる」ため、
# アップロード中にジョブが強制終了されるとDBが消えたままになる事故が起きた。
# 新しいDBを別名(boatrace-new.db)で上げきってから、名前の付け替えで差し替える。
# 直前の版は boatrace-prev.db として1つ残す。
# 使い方: bash scripts/upload_db.sh <DBファイル> <owner/repo>
set -euo pipefail
f="$1"; repo="$2"
python3 - "$f" <<'PY'
import os, sqlite3, sys
f = sys.argv[1]
size = os.path.getsize(f) if os.path.exists(f) else 0
if size < 100_000_000:
    sys.exit(f"DBが小さすぎる/存在しない({size} bytes)ためアップロードしない")
if sqlite3.connect(f).execute("pragma quick_check").fetchone()[0] != "ok":
    sys.exit("DBの整合性チェックに失敗したためアップロードしない")
PY
tmpdir=$(mktemp -d); cp "$f" "$tmpdir/boatrace-new.db"
gh release upload db-latest "$tmpdir/boatrace-new.db" --repo "$repo" --clobber
rm -rf "$tmpdir"
assets=$(gh api "repos/$repo/releases/tags/db-latest" --jq '.assets[]|"\(.name) \(.id)"')
id_of() { echo "$assets" | awk -v n="$1" '$1==n{print $2}'; }
new=$(id_of boatrace-new.db); cur=$(id_of boatrace.db); prev=$(id_of boatrace-prev.db)
[ -n "$new" ] || { echo "boatrace-new.db が見つからない"; exit 1; }
[ -n "$prev" ] && gh api -X DELETE "repos/$repo/releases/assets/$prev" >/dev/null
[ -n "$cur" ] && gh api -X PATCH "repos/$repo/releases/assets/$cur" -f name=boatrace-prev.db >/dev/null
gh api -X PATCH "repos/$repo/releases/assets/$new" -f name=boatrace.db >/dev/null
echo "DBをアップロードしました"
