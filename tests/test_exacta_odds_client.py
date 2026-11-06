from pathlib import Path

from boatrace_predictor.data.exacta_odds_client import parse_exacta_odds

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
