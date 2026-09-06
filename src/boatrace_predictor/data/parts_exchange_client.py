"""ボートレース公式サイトの直前情報ページから部品交換情報をPlaywrightで取得する。

Boatrace Open APIのpreview(直前情報)にはチルト・展示タイム等はあるが、
部品交換(モーター整備での部品交換)は含まれていない。5点改善計画の項目3
(2026-08-23): 直近パーツ交換は選手側の「今節のモーターに不安/期待がある」
シグナルになりうるという仮説だが、過去に遡って取得できるデータではない
(公式サイトが直前情報を残すのは開催中のみ)ため、実際に予測精度に効くかは
これから溜まるデータで検証する。現時点ではまだNUMERIC_FEATURESには入れていない。

ページ構造(2026-08-23、boatrace.jp/owpc/pc/race/beforeinfo で確認):
出走表テーブル(class="is-w748")の1艇分は
「N\t\t選手名\t体重kg\t展示タイム\tチルト\tプロペラ欄\t\n(部品交換の行、なければ次はタブ始まり)...」
という並びのinnerTextになる。部品交換の行は「リング×２」のように
部品交換凡例(ピストン/リング/電気/キャブ/シリンダ/シャフト/ギヤ/キャリボ)の
短縮名を含むテキストで、交換が無ければその行自体が現れない。
プロペラ交換時はプロペラ欄が「新」になる。
"""

import re
from datetime import date

from playwright.sync_api import sync_playwright

_PART_KEYWORDS = ["ピストン", "リング", "電気", "キャブ", "シリンダ", "シャフト", "ギヤ", "キャリボ"]
_ROW_RE = re.compile(
    r"([1-6])\t\t[^\t\n]+\t[\d.]+kg\t[\d.]+\t[-\d.]+\t([^\t\n]*)\t\n([^\n]*)"
)


def _parse_parts_exchange_text(table_text: str) -> dict[int, list[str]]:
    """出走表テーブルのinnerTextから艇ごとの交換部品名リストを抜き出す。

    交換が無い艇はキー自体を作らない(空リストとは区別する)。
    """
    blocks = re.split(r"\n(?=[1-6]\t\t)", table_text)
    result: dict[int, list[str]] = {}
    for block in blocks:
        m = _ROW_RE.match(block)
        if not m:
            continue
        boat = int(m.group(1))
        propeller_field = m.group(2).strip()
        next_line = m.group(3)

        parts = []
        if propeller_field == "新":
            parts.append("プロペラ")
        for kw in _PART_KEYWORDS:
            if kw in next_line:
                parts.append(kw)
        if parts:
            result[boat] = parts
    return result


class PartsExchangeClient:
    """ブラウザを1つだけ起動して使い回す(レースごとに起動すると遅い)。

    browser を渡すとそれを使い回す(OddsClientと共有する場合など)。省略時は
    自分でPlaywrightを起動する。同一プロセスでsync_playwright()を複数回起動すると
    エラーになるため(data/odds_client.py参照)、併用時は呼び出し側で共有すること。
    """

    def __init__(self, browser=None) -> None:
        self._playwright = None
        self._browser = browser
        self._owns_browser = browser is None

    def __enter__(self) -> "PartsExchangeClient":
        if self._owns_browser:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._owns_browser:
            if self._browser is not None:
                self._browser.close()
            if self._playwright is not None:
                self._playwright.stop()

    def fetch_parts_exchange(
        self, stadium_number: int, target_date: date, race_number: int
    ) -> dict[int, list[str]]:
        assert self._browser is not None, "with PartsExchangeClient() as client: の中で使うこと"
        url = (
            "https://www.boatrace.jp/owpc/pc/race/beforeinfo"
            f"?rno={race_number}&jcd={stadium_number:02d}&hd={target_date:%Y%m%d}"
        )
        page = self._browser.new_page()
        try:
            page.goto(url, wait_until="load", timeout=45000)
            page.wait_for_timeout(1500)
            table = page.locator("table.is-w748").first
            if table.count() == 0:
                return {}
            text = table.inner_text()
        finally:
            page.close()
        return _parse_parts_exchange_text(text)
