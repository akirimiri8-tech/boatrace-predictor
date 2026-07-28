import json
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from boatrace_predictor.data.schemas import ApiV1Response
from boatrace_predictor.features.dataset import build_dataset, build_race_features
from boatrace_predictor.persistence.db import save_race
from boatrace_predictor.persistence.models import Race

FIXTURES = Path(__file__).parent / "fixtures"


def _engine_with_fixture_data():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    raw = json.loads(
        (FIXTURES / "api_v1_20260720_trimmed.json").read_text(encoding="utf-8")
    )
    parsed = ApiV1Response.model_validate(raw)
    with Session(engine) as session:
        for stadium in parsed.programs.stadiums.values():
            for race in stadium.races.values():
                save_race(session, race)
        session.commit()
    return engine


def test_build_race_features_has_six_boats_and_expected_columns() -> None:
    engine = _engine_with_fixture_data()
    with Session(engine) as session:
        # stadium_number=2, race_number=1 は全艇正常着順であることを確認済みのレース
        race_id = session.exec(
            select(Race.id).where(Race.stadium_number == 2, Race.number == 1)
        ).first()
        df = build_race_features(session, race_id)

    assert len(df) == 6
    assert set(df["racer_boat_number"]) == {1, 2, 3, 4, 5, 6}
    assert df["is_win"].sum() == 1  # 正常終了なら1着はちょうど1艇
    # 展示タイムの順位は1〜6が重複なく振られているはず
    ranks = sorted(df["exhibition_time_rank"].dropna().tolist())
    assert ranks == [1, 2, 3, 4, 5, 6]


def test_build_dataset_concatenates_all_races() -> None:
    engine = _engine_with_fixture_data()
    with Session(engine) as session:
        df = build_dataset(session)

    assert len(df) % 6 == 0
    assert len(df) > 6  # フィクスチャは複数レース含む
