"""DBエンジン/セッションとデータ保存ロジック。"""

from sqlmodel import Session, SQLModel, create_engine, select

from boatrace_predictor.config import settings
from boatrace_predictor.data.schemas import RaceProgram
from boatrace_predictor.persistence.models import (
    Payout,
    PreviewEntry,
    PreviewWeather,
    Race,
    RacerEntry,
    RaceWeather,
    ResultEntry,
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


def save_race(session: Session, race: RaceProgram) -> None:
    """1レース分(出走表+直前情報+結果)をまとめて保存する。

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
