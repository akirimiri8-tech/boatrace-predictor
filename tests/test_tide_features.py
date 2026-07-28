import json

from sqlmodel import Session, SQLModel, create_engine

from boatrace_predictor.features.dataset import _tide_level_at, _tide_trend_at, build_race_features
from boatrace_predictor.persistence.models import Race, RacerEntry, TideDaily


def test_tide_level_at_interpolates_between_points() -> None:
    curve = [
        {"time": "10:00", "cm": 100.0},
        {"time": "10:20", "cm": 120.0},
    ]
    assert _tide_level_at(curve, 10 * 60) == 100.0
    assert _tide_level_at(curve, 10 * 60 + 20) == 120.0
    assert _tide_level_at(curve, 10 * 60 + 10) == 110.0  # 中間点


def test_tide_level_at_clamps_outside_range() -> None:
    curve = [{"time": "10:00", "cm": 100.0}, {"time": "10:20", "cm": 120.0}]
    assert _tide_level_at(curve, 0) == 100.0
    assert _tide_level_at(curve, 23 * 60) == 120.0


def test_tide_trend_positive_when_rising() -> None:
    curve = [
        {"time": "09:30", "cm": 90.0},
        {"time": "10:00", "cm": 100.0},
        {"time": "10:30", "cm": 110.0},
    ]
    trend = _tide_trend_at(curve, 10 * 60, window_minutes=60)
    assert trend == 20.0  # (110-90) / 1時間


def test_build_race_features_includes_tide_columns_when_available() -> None:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    curve = [{"time": f"{h:02d}:00", "cm": 100.0 + h} for h in range(0, 25, 1)]

    with Session(engine) as session:
        race = Race(date="2026-07-20", stadium_number=24, number=1, closed_at="2026-07-20 10:00:00")
        session.add(race)
        session.flush()
        for boat in range(1, 7):
            session.add(
                RacerEntry(
                    race_id=race.id,
                    racer_boat_number=boat,
                    racer_name=f"racer{boat}",
                    racer_number=1000 + boat,
                )
            )
        session.add(
            TideDaily(
                date="2026-07-20",
                stadium_number=24,
                tide_name="大潮",
                moon_age=5.7,
                curve_json=json.dumps(curve),
            )
        )
        session.commit()

        df = build_race_features(session, race.id)

    assert (df["tide_level_cm"] == 110.0).all()  # 10:00時点のcm
    assert (df["tide_name"] == "大潮").all()
    assert (df["moon_age"] == 5.7).all()
