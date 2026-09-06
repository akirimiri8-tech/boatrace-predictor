from sqlmodel import Session, SQLModel, create_engine, select

from boatrace_predictor.data.parts_exchange_client import _parse_parts_exchange_text
from boatrace_predictor.persistence.db import save_parts_exchange
from boatrace_predictor.persistence.models import PartsExchange, Race

# 2026-08-23に実際に取得できた直前情報ページのtable.is-w748.innerText()を再現したもの
# (中日カップ 常滑12R、5号艇・6号艇がピストンリングを交換していたレース)
SAMPLE_TEXT = (
    "枠\t写真\tボートレーサー\t体重\t展示\nタイム\tチルト\tプロペラ\t部品交換\t前走成績\n調整重量\n"
    "1\t\t赤岩　　善生\t54.6kg\t6.76\t-0.5\t \t\n\tR\t7\n進入\t2\n0.0\tST\t.17\n着順\t４\n"
    "2\t\t渡辺　　浩司\t52.1kg\t6.68\t0.0\t \t\n\tR\t6\n進入\t1\n0.0\tST\t.12\n着順\t１\n"
    "3\t\t渡邉　　和将\t50.5kg\t6.78\t0.0\t \t\n\tR\t5\n進入\t4\n1.5\tST\t.18\n着順\t３\n"
    "4\t\t黒野　　元基\t52.0kg\t6.75\t0.0\t \t\n\tR\t8\n進入\t6\n0.0\tST\t.16\n着順\t３\n"
    "5\t\t川原　　祐明\t52.0kg\t6.77\t0.0\t \t\nリング×２\n\tR\t3\n進入\t3\n0.0\tST\t.25\n着順\t４\n"
    "6\t\t今泉　　友吾\t52.0kg\t6.78\t0.0\t \t\nリング×２\n\tR\t4\n進入\t2\n0.0\tST\t.14\n着順\t３"
)


def test_parse_parts_exchange_text_extracts_ring_exchange() -> None:
    result = _parse_parts_exchange_text(SAMPLE_TEXT)
    assert result == {5: ["リング"], 6: ["リング"]}


def test_parse_parts_exchange_text_detects_propeller_change() -> None:
    text = "1\t\t選手　　太郎\t52.0kg\t6.80\t0.0\t新\t\n\tR\t1\n進入\t1\n0.0\tST\t.10\n着順\t１"
    result = _parse_parts_exchange_text(text)
    assert result == {1: ["プロペラ"]}


def test_parse_parts_exchange_text_no_match_returns_empty_dict() -> None:
    assert _parse_parts_exchange_text("ページが見つかりません") == {}


def test_save_parts_exchange_persists_and_replaces() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        race = Race(date="2026-08-23", stadium_number=8, number=12)
        session.add(race)
        session.flush()

        parts = _parse_parts_exchange_text(SAMPLE_TEXT)
        save_parts_exchange(session, race.id, "2026-08-23T11:00:00", parts)
        session.commit()

        rows = session.exec(
            select(PartsExchange).where(PartsExchange.race_id == race.id)
        ).all()
        assert len(rows) == 2
        boat5 = next(r for r in rows if r.racer_boat_number == 5)
        assert boat5.parts == "リング"

        # 再取得で置き換わる(重複しない)
        save_parts_exchange(session, race.id, "2026-08-23T11:20:00", {1: ["プロペラ"]})
        session.commit()
        rows2 = session.exec(
            select(PartsExchange).where(PartsExchange.race_id == race.id)
        ).all()
        assert len(rows2) == 1
        assert rows2[0].racer_boat_number == 1
        assert rows2[0].parts == "プロペラ"
