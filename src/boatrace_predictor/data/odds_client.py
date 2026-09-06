"""ボートレース公式サイトのオッズページをPlaywrightでスクレイピングする。

Boatrace Open APIには事前オッズが含まれていない(結果確定後の払戻金しか
分からない)。オッズページはJavaScriptで動的描画されるため、単純な
HTTPリクエストでは中身が取れず、Playwrightでブラウザを実際に操作して
取得する(2026-08-22、動作確認済み)。

公式サイトへの直接アクセスになるため、Boatrace Open API(GitHub Pages
ミラー)より負荷・規約面のリスクが高い。アクセス頻度は必要最小限に
抑えること(1レースにつき1回程度)。
"""

from datetime import date

from playwright.sync_api import sync_playwright


def _parse_odds_text(text: str) -> tuple[dict[int, float], dict[int, tuple[float, float]]]:
    """オッズページのinner_text(body)から単勝・複勝オッズを抜き出す。

    ページ構造(2026-08-22時点):
    「単勝オッズ」見出し→ヘッダ行→1〜6の艇番・選手名・オッズが6行→「複勝オッズ」見出し→…
    """
    lines = [line.strip() for line in text.split("\n")]

    def rows_between(start_label: str, end_label: str | None) -> list[str]:
        try:
            start = lines.index(start_label)
        except ValueError:
            return []
        collected = []
        for line in lines[start + 1 :]:
            if end_label is not None and line == end_label:
                break
            collected.append(line)
            if len(collected) > 20:  # 異常な構造でも無限に伸びないための安全弁
                break
        return collected

    win_odds: dict[int, float] = {}
    for line in rows_between("単勝オッズ", "複勝オッズ"):
        parts = [p for p in line.split("\t") if p.strip()]
        if len(parts) >= 2 and parts[0].isdigit():
            boat = int(parts[0])
            if 1 <= boat <= 6:
                try:
                    win_odds[boat] = float(parts[-1])
                except ValueError:
                    pass

    place_odds: dict[int, tuple[float, float]] = {}
    for line in rows_between("複勝オッズ", None):
        if len(place_odds) >= 6:
            break
        parts = [p for p in line.split("\t") if p.strip()]
        if len(parts) >= 2 and parts[0].isdigit():
            boat = int(parts[0])
            odds_str = parts[-1]
            if 1 <= boat <= 6 and "-" in odds_str:
                lo, hi = odds_str.split("-", 1)
                try:
                    place_odds[boat] = (float(lo), float(hi))
                except ValueError:
                    pass

    return win_odds, place_odds


class OddsClient:
    """ブラウザを1つだけ起動して使い回す(レースごとに起動すると遅い)。

    browser を渡すとそれを使い回す(他のクライアントと共有する場合)。省略時は
    自分でPlaywrightを起動する。同一プロセス内でsync_playwright()を複数回
    起動すると "using Playwright Sync API inside the asyncio loop" エラーに
    なることがある(2026-08-24、daily_predict.pyでOddsClientとPartsExchangeClientを
    両方使ったときに発生・判明)ため、複数クライアントを併用する場合は
    呼び出し側で1つのbrowserを起動して共有すること。
    """

    def __init__(self, browser=None) -> None:
        self._playwright = None
        self._browser = browser
        self._owns_browser = browser is None

    def __enter__(self) -> "OddsClient":
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

    def fetch_odds(
        self, stadium_number: int, target_date: date, race_number: int
    ) -> tuple[dict[int, float], dict[int, tuple[float, float]]]:
        assert self._browser is not None, "with OddsClient() as client: の中で使うこと"
        url = (
            "https://www.boatrace.jp/owpc/pc/race/oddstf"
            f"?rno={race_number}&jcd={stadium_number:02d}&hd={target_date:%Y%m%d}"
        )
        page = self._browser.new_page()
        try:
            page.goto(url, wait_until="load", timeout=45000)
            page.wait_for_timeout(1500)
            text = page.inner_text("body")
        finally:
            page.close()
        return _parse_odds_text(text)
