from sqlmodel import Session, SQLModel, create_engine, select

from boatrace_predictor.data.odds_client import _parse_odds_text
from boatrace_predictor.persistence.db import save_odds
from boatrace_predictor.persistence.models import Odds, Race

# 2026-08-22に実際に取得できたオッズページのinner_text(body)を再現したもの
SAMPLE_TEXT = """HOME
オッズ更新時間 10:58

更新

単勝オッズ
 	ボートレーサー	単勝オッズ
1	大塚　　雅治	2.0
2	川辺　　郭人	18.0
3	重富　　伸也	18.0
4	多田　　有佑	18.0
5	野田　　昇吾	1.5
6	栗原　　一馬	0.0
複勝オッズ
 	ボートレーサー	複勝オッズ
1	大塚　　雅治	1.0-1.5
2	川辺　　郭人	1.3-3.5
3	重富　　伸也	1.6-4.7
4	多田　　有佑	0.0-0.0
5	野田　　昇吾	1.6-4.7
6	栗原　　一馬	0.0-0.0
ボートレースガイドはこちら
"""


def test_parse_odds_text_extracts_win_odds() -> None:
    win_odds, _ = _parse_odds_text(SAMPLE_TEXT)
    assert win_odds == {1: 2.0, 2: 18.0, 3: 18.0, 4: 18.0, 5: 1.5, 6: 0.0}


def test_parse_odds_text_extracts_place_odds_range() -> None:
    _, place_odds = _parse_odds_text(SAMPLE_TEXT)
    assert place_odds[1] == (1.0, 1.5)
    assert place_odds[3] == (1.6, 4.7)
    assert len(place_odds) == 6


def test_parse_odds_text_handles_missing_sections() -> None:
    win_odds, place_odds = _parse_odds_text("ページが見つかりません")
    assert win_odds == {}
    assert place_odds == {}


def test_save_odds_persists_and_replaces() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        race = Race(date="2026-08-22", stadium_number=3, number=1)
        session.add(race)
        session.flush()

        win_odds, place_odds = _parse_odds_text(SAMPLE_TEXT)
        save_odds(session, race.id, "2026-08-22T10:58:00", win_odds, place_odds)
        session.commit()

        rows = session.exec(select(Odds).where(Odds.race_id == race.id)).all()
        assert len(rows) == 6
        boat1 = next(r for r in rows if r.racer_boat_number == 1)
        assert boat1.win_odds == 2.0
        assert boat1.place_odds_low == 1.0
        assert boat1.place_odds_high == 1.5

        # 再取得で置き換わる(重複しない)
        save_odds(session, race.id, "2026-08-22T11:10:00", {1: 2.2}, {})
        session.commit()
        rows2 = session.exec(select(Odds).where(Odds.race_id == race.id)).all()
        assert len(rows2) == 1
        assert rows2[0].win_odds == 2.2
