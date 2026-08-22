"""DBエンジン/セッションとデータ保存ロジック。"""

import json

from sqlmodel import Session, SQLModel, create_engine, select

from boatrace_predictor.config import settings
from boatrace_predictor.data.schemas import RaceProgram
from boatrace_predictor.data.tide_schemas import DayChart
from boatrace_predictor.persistence.models import (
    Odds,
    Payout,
    Prediction,
    PredictionScore,
    PreviewEntry,
    PreviewWeather,
    Race,
    RacerEntry,
    RaceWeather,
    ResultEntry,
    TideDaily,
)

settings.db_path.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{settings.db_path}")


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def _get_or_create_race(session: Session, date: str, stadium_number: int, number: int) -> Race:
    race = session.exec(
        select(Race).where(
            Race.date == date,
            Race.stadium_number == stadium_number,
            Race.number == number,
        )
    ).first()
    if race is None:
        race = Race(date=date, stadium_number=stadium_number, number=number)
        session.add(race)
        session.flush()  # race.id を確定させる
    return race


def _replace(session: Session, model: type, race_id: int) -> None:
    existing = session.exec(select(model).where(model.race_id == race_id)).all()
    for row in existing:
        session.delete(row)
    session.flush()


def save_race(session: Session, race: RaceProgram) -> int:
    """1レース分(出走表+直前情報+結果)をまとめて保存し、race_idを返す。

    既存データがあれば置き換える(番組変更・直前情報確定・結果確定のたびに
    同じレースを再取得して上書きする運用を想定)。
    """
    db_race = _get_or_create_race(session, race.date, race.stadium_number, race.race_number)
    db_race.closed_at = race.closed_at
    db_race.day_number = race.day_number
    db_race.grade_number = race.grade_number
    db_race.title = race.title
    db_race.subtitle = race.subtitle
    db_race.distance = race.distance
    session.add(db_race)
    session.flush()
    race_id = db_race.id
    assert race_id is not None

    if race.racers:
        _replace(session, RacerEntry, race_id)
        for r in race.racers.values():
            session.add(
                RacerEntry(
                    race_id=race_id,
                    racer_boat_number=r.entry_number,
                    racer_name=r.name,
                    racer_number=r.number,
                    racer_class_number=r.rank_number,
                    racer_branch_number=r.branch_number,
                    racer_birthplace_number=r.birthplace_number,
                    racer_age=r.age,
                    racer_weight=r.weight,
                    racer_flying_count=r.flying_count,
                    racer_late_count=r.late_count,
                    racer_average_start_timing=r.average_start_timing,
                    racer_national_top_1_percent=r.national_win_rate,
                    racer_national_top_2_percent=r.national_top_2_percent,
                    racer_national_top_3_percent=r.national_top_3_percent,
                    racer_local_top_1_percent=r.local_win_rate,
                    racer_local_top_2_percent=r.local_top_2_percent,
                    racer_local_top_3_percent=r.local_top_3_percent,
                    racer_assigned_motor_number=r.motor_number,
                    racer_assigned_motor_top_2_percent=r.motor_top_2_percent,
                    racer_assigned_motor_top_3_percent=r.motor_top_3_percent,
                    racer_assigned_boat_number=r.boat_number,
                    racer_assigned_boat_top_2_percent=r.boat_top_2_percent,
                    racer_assigned_boat_top_3_percent=r.boat_top_3_percent,
                )
            )

    if race.preview is not None:
        existing_weather = session.exec(
            select(PreviewWeather).where(PreviewWeather.race_id == race_id)
        ).first()
        if existing_weather:
            session.delete(existing_weather)
        _replace(session, PreviewEntry, race_id)
        session.flush()

        p = race.preview
        session.add(
            PreviewWeather(
                race_id=race_id,
                wind_speed=p.wind_speed,
                wind_direction_number=p.wind_direction_number,
                wave_height=p.wave_height,
                weather_number=p.weather_number,
                air_temperature=p.air_temperature,
                water_temperature=p.water_temperature,
            )
        )
        for pr in p.racers.values():
            session.add(
                PreviewEntry(
                    race_id=race_id,
                    racer_boat_number=pr.entry_number,
                    course_number=pr.course_number,
                    start_timing=pr.start_timing,
                    weight=pr.weight,
                    weight_adjustment=pr.weight_adjustment,
                    exhibition_time=pr.exhibition_time,
                    tilt_adjustment=pr.tilt_adjustment,
                )
            )

    if race.result is not None:
        existing_weather = session.exec(
            select(RaceWeather).where(RaceWeather.race_id == race_id)
        ).first()
        if existing_weather:
            session.delete(existing_weather)
        _replace(session, ResultEntry, race_id)
        _replace(session, Payout, race_id)
        session.flush()

        res = race.result
        session.add(
            RaceWeather(
                race_id=race_id,
                wind_speed=res.wind_speed,
                wind_direction_number=res.wind_direction_number,
                wave_height=res.wave_height,
                weather_number=res.weather_number,
                air_temperature=res.air_temperature,
                water_temperature=res.water_temperature,
                technique_number=res.technique_number,
            )
        )
        for rr in res.racers.values():
            session.add(
                ResultEntry(
                    race_id=race_id,
                    racer_boat_number=rr.entry_number,
                    racer_course_number=rr.course_number,
                    racer_start_timing=rr.start_timing,
                    racer_place_number=rr.place_number,
                    racer_number=rr.number,
                    racer_name=rr.name,
                )
            )
        if res.payouts is not None:
            for bet_type, entries in res.payouts.model_dump().items():
                for entry in entries:
                    session.add(
                        Payout(
                            race_id=race_id,
                            bet_type=bet_type,
                            combination=entry["combination"],
                            amount=entry["amount"],
                        )
                    )

    return race_id


def save_tide_day(session: Session, date: str, stadium_number: int, chart: DayChart) -> None:
    existing = session.exec(
        select(TideDaily).where(
            TideDaily.date == date, TideDaily.stadium_number == stadium_number
        )
    ).first()
    if existing:
        session.delete(existing)
        session.flush()

    session.add(
        TideDaily(
            date=date,
            stadium_number=stadium_number,
            tide_name=chart.moon.title,
            moon_age=chart.moon.age,
            curve_json=json.dumps([p.model_dump() for p in chart.tide]),
        )
    )


def save_prediction(
    session: Session,
    race_id: int,
    predicted_at: str,
    model_name: str,
    ranked_boats: list[int],
    scores: dict[int, float],
    features_snapshot: list[dict] | None = None,
) -> Prediction:
    """レース1件分の予想をログに保存する。同モデル・同レースの古い予想は削除して置き換える。

    features_snapshot: 予想に使った6艇分の生の特徴量(監査用)。省略可。
    """
    existing = session.exec(
        select(Prediction).where(
            Prediction.race_id == race_id, Prediction.model_name == model_name
        )
    ).all()
    for old in existing:
        old_scores = session.exec(
            select(PredictionScore).where(PredictionScore.prediction_id == old.id)
        ).all()
        for s in old_scores:
            session.delete(s)
        session.delete(old)
    session.flush()

    prediction = Prediction(
        race_id=race_id,
        predicted_at=predicted_at,
        model_name=model_name,
        predicted_1st=ranked_boats[0],
        predicted_2nd=ranked_boats[1] if len(ranked_boats) > 1 else None,
        predicted_3rd=ranked_boats[2] if len(ranked_boats) > 2 else None,
        features_json=json.dumps(features_snapshot) if features_snapshot is not None else None,
    )
    session.add(prediction)
    session.flush()

    for boat, score in scores.items():
        session.add(
            PredictionScore(prediction_id=prediction.id, racer_boat_number=boat, score=score)
        )
    return prediction


def save_odds(
    session: Session,
    race_id: int,
    scraped_at: str,
    win_odds: dict[int, float],
    place_odds: dict[int, tuple[float, float]],
) -> None:
    """発走前オッズを保存する。同じレースの古いオッズは削除して置き換える。"""
    existing = session.exec(select(Odds).where(Odds.race_id == race_id)).all()
    for old in existing:
        session.delete(old)
    session.flush()

    boats = set(win_odds) | set(place_odds)
    for boat in boats:
        lo_hi = place_odds.get(boat)
        session.add(
            Odds(
                race_id=race_id,
                scraped_at=scraped_at,
                racer_boat_number=boat,
                win_odds=win_odds.get(boat),
                place_odds_low=lo_hi[0] if lo_hi else None,
                place_odds_high=lo_hi[1] if lo_hi else None,
            )
        )
