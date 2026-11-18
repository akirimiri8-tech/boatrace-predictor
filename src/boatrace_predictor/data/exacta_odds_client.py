"""公式サイトの2連単オッズ表(全30通り)を取得する。

oddstf(単勝・複勝)と違い、odds2tf は静的HTMLなのでPlaywright不要(2026-11-07確認)。
締切前の複数時点で記録し、確定オッズとの差を見るために使う(OddsSnapshotの2連単版)。
"""

import os
import time
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any

import httpx

_URL = "https://www.boatrace.jp/owpc/pc/race/odds2tf"
_HEADERS = {"User-Agent": "boatrace-research-personal/1.0"}


class _Tables(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.out: list[dict] = []
        self._cur: dict | None = None
        self._row: list | None = None
        self._cell: list | None = None
        self._in_head = False
        self._in_label = False
        self._label = ""
        self._last_label: str | None = None

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "") or ""
        if tag == "span" and "title7_mainLabel" in cls:
            self._in_label, self._label = True, ""
        if tag == "table":
            self._cur = {"label": self._last_label, "head": [], "body": []}
        if tag == "thead":
            self._in_head = True
        if tag == "tr" and self._cur is not None:
            self._row = []
        if tag in ("td", "th") and self._row is not None:
            self._cell = [cls, ""]

    def handle_endtag(self, tag):
        if tag == "span" and self._in_label:
            self._in_label, self._last_label = False, self._label
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(tuple(self._cell))
            self._cell = None
        if tag == "tr" and self._row is not None and self._cur is not None:
            (self._cur["head"] if self._in_head else self._cur["body"]).append(self._row)
            self._row = None
        if tag == "thead":
            self._in_head = False
        if tag == "table" and self._cur is not None:
            self.out.append(self._cur)
            self._cur = None

    def handle_data(self, data):
        if self._in_label:
            self._label += data
        if self._cell is not None:
            self._cell[1] += data


def parse_exacta_odds(html: str) -> dict[str, float]:
    """{'1-2': 3.5, ...}。キーは『1着-2着』。取れなかった組合せ(欠場等)は含めない。
    表は『列見出し=1着の艇、行ラベル=2着の艇』の並び。2026-11-07、大村7R(11/4)の実際の
    2連単払戻(1-3 = 5.7倍)との照合で確認した(逆に読むと53.6倍になり食い違う)。"""
    parser = _Tables()
    parser.feed(html)
    odds: dict[str, float] = {}
    for t in parser.out:
        if "2連単" not in (t["label"] or "") or not t["head"]:
            continue
        cols = [int(c[1]) for c in t["head"][0][0::2]]
        for row in t["body"]:
            for j, col in enumerate(cols):
                if 2 * j + 1 >= len(row):
                    break
                label, val = row[2 * j], row[2 * j + 1]
                if "oddsPoint" in val[0] and label[1].strip().isdigit():
                    try:
                        odds[f"{col}-{int(label[1])}"] = float(val[1].strip())
                    except ValueError:
                        pass
    return odds


# 2026-11-18: 2連単オッズは自前のオッズ配信API(別システム)から読めるようにした。
# 接続先と合言葉は環境変数で渡す(コードやリポジトリには置かない)。未設定なら従来どおり直接取得する。
_FEED_URL_ENV = "ODDS_FEED_URL"
_FEED_TOKEN_ENV = "ODDS_FEED_TOKEN"
# 配信側は締切前の決まった時点で記録する。呼び出し側は「いま」のオッズとして時刻を付けて保存するので、
# 古い記録を返すと別時点のオッズが紛れ込む。新しい記録だけを採用し、まだ無ければ少し待つ。
_FEED_MAX_AGE_SECONDS = 60.0
_FEED_WAIT_SECONDS = 30.0
_FEED_POLL_SECONDS = 3.0
_JST = timezone(timedelta(hours=9))


def _latest_fresh_feed_odds(snapshots: list[dict[str, Any]], now: datetime, max_age: float) -> dict[str, float]:
    """配信APIの記録のうち、max_age秒以内に取られた最新のものを {'1-2': 3.5, ...} で返す。無ければ空。"""
    best: tuple[datetime, dict[str, Any]] | None = None
    for snap in snapshots:
        captured = datetime.fromisoformat(snap["captured_at"])
        if (now - captured).total_seconds() > max_age:
            continue
        if best is None or captured > best[0]:
            best = (captured, snap["odds"])
    if best is None:
        return {}
    return {k: float(v) for k, v in best[1].items() if v is not None}


def _fetch_from_feed(base_url: str, token: str, stadium_number: int, race_date: str, race_number: int) -> dict[str, float]:
    params = {"date": race_date.replace("-", ""), "venue": stadium_number, "race": race_number, "type": "exacta"}
    deadline = time.monotonic() + _FEED_WAIT_SECONDS
    with httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=15) as client:
        while True:
            resp = client.get(base_url.rstrip("/") + "/v1/odds", params=params)
            resp.raise_for_status()
            odds = _latest_fresh_feed_odds(resp.json()["snapshots"], datetime.now(_JST), _FEED_MAX_AGE_SECONDS)
            if odds or time.monotonic() >= deadline:
                return odds
            time.sleep(_FEED_POLL_SECONDS)


def fetch_exacta_odds(stadium_number: int, race_date: str, race_number: int, client: httpx.Client | None = None) -> dict[str, float]:
    """race_date は 'YYYY-MM-DD' または 'YYYYMMDD'。"""
    feed_url, feed_token = os.environ.get(_FEED_URL_ENV), os.environ.get(_FEED_TOKEN_ENV)
    if feed_url and feed_token:
        return _fetch_from_feed(feed_url, feed_token, stadium_number, race_date, race_number)
    params = {"rno": race_number, "jcd": f"{stadium_number:02d}", "hd": race_date.replace("-", "")}
    own = client is None
    client = client or httpx.Client(headers=_HEADERS, timeout=30)
    try:
        resp = client.get(_URL, params=params)
        # 2026-11-18: 公式サイトがこのクライアントに403(Access Denied)を返すようになったのに、
        # 空の結果として黙って処理されて気づくのが遅れた。失敗は例外にしてログに残す。
        resp.raise_for_status()
        return parse_exacta_odds(resp.text)
    finally:
        if own:
            client.close()
