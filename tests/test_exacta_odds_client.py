from datetime import datetime, timedelta, timezone
from pathlib import Path

from boatrace_predictor.data.exacta_odds_client import _pick_feed_odds, parse_exacta_odds

_FIXTURE = Path(__file__).parent / "fixtures" / "odds2tf_exacta_table.html"


def test_parses_all_30_exacta_combinations() -> None:
    odds = parse_exacta_odds(_FIXTURE.read_text(encoding="utf-8"))
    assert len(odds) == 30
    assert all(a != b for a, b in (k.split("-") for k in odds))  # 同じ艇の組はない
    assert all(v > 0 for v in odds.values())


def test_keys_are_first_then_second_place() -> None:
    odds = parse_exacta_odds(_FIXTURE.read_text(encoding="utf-8"))
    assert {"1-2", "2-1", "6-5"} <= set(odds)
    # 人気は1号艇頭に偏るので、1号艇が頭の並びの平均オッズは他より低いはず
    head1 = [v for k, v in odds.items() if k.startswith("1-")]
    head6 = [v for k, v in odds.items() if k.startswith("6-")]
    assert sum(head1) / len(head1) < sum(head6) / len(head6)


def test_direction_matches_real_payout() -> None:
    # 実ページ(2025-10-01 桐生/浜名湖系の1R)の1番人気は1号艇頭になるはず。向きを逆に読むと、
    # 1号艇が2着の並び(例: 3-1)が人気上位になってしまう。実際の払戻との照合で確認済みの向き。
    odds = parse_exacta_odds(_FIXTURE.read_text(encoding="utf-8"))
    favorite = min(odds, key=odds.get)
    assert favorite.startswith("1-"), favorite


def test_returns_empty_for_unrelated_html() -> None:
    assert parse_exacta_odds("<html><body>no table</body></html>") == {}


_NOW = datetime(2026, 11, 18, 19, 24, 30, tzinfo=timezone(timedelta(hours=9)))


def _snap(label: str, captured_at: str, odds: dict) -> dict:
    return {"label": label, "captured_at": captured_at, "odds": odds}


def test_feed_picks_snapshot_for_requested_time_point() -> None:
    snaps = [
        _snap("12m", "2026-11-18T19:12:00+09:00", {"1-2": 3.0}),
        _snap("7m", "2026-11-18T19:17:00+09:00", {"1-2": 2.8, "1-3": None}),
    ]
    # 欠場などで値の無い並びは含めない(直接取得のときと同じ扱い)
    assert _pick_feed_odds(snaps, _NOW, 7) == {"1-2": 2.8}
    # 見回りが窓の後半に来ても、その時点の記録なら古くても使う
    assert _pick_feed_odds(snaps, _NOW, 12) == {"1-2": 3.0}


def test_feed_returns_empty_when_time_point_not_recorded_yet() -> None:
    # 2分前の記録がまだ無いのに、7分前の記録を2分前として返さないこと
    snaps = [_snap("7m", "2026-11-18T19:24:00+09:00", {"1-2": 2.8})]
    assert _pick_feed_odds(snaps, _NOW, 2) == {}


def test_feed_without_time_point_uses_only_fresh_snapshot() -> None:
    fresh = [_snap("7m", "2026-11-18T19:24:00+09:00", {"1-2": 2.8})]
    stale = [_snap("7m", "2026-11-18T19:19:00+09:00", {"1-2": 3.0})]
    assert _pick_feed_odds(fresh, _NOW, None) == {"1-2": 2.8}
    assert _pick_feed_odds(stale, _NOW, None) == {}
