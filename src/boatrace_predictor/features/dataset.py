"""DB上のレースデータを1艇1行のテーブル(pandas DataFrame)に変換する。

同一レース内の6艇を横断した相対特徴量(展示タイム順位など)も含めて、
予想モデル・バックテストの両方から使えるデータセットを作る。

着順(racer_place_number)は 1〜6 が正常着順、それ以外(None やフライング/転覆/
エンジン故障などの番号)は異常着順として扱う。異常着順は is_win/is_top2/is_top3を
すべて0にし、is_abnormal_finish で区別できるようにしている。
"""

import json
from datetime import datetime

import pandas as pd
from sqlmodel import Session, select

from boatrace_predictor.config import STADIUM_WATER_TYPE
from boatrace_predictor.persistence.models import (
    Payout,
    PreviewEntry,
    PreviewWeather,
    Race,
    RacerEntry,
    RaceWeather,
    ResultEntry,
    TideDaily,
)

_NORMAL_PLACES = {1, 2, 3, 4, 5, 6}


def _parse_hhmm_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _tide_level_at(curve: list[dict], minutes: float) -> float | None:
    """潮位カーブ(20分間隔)から線形補間で任意時刻の潮位(cm)を求める。"""
    if not curve:
        return None
    points = sorted((_parse_hhmm_minutes(p["time"]), p["cm"]) for p in curve)
    if minutes <= points[0][0]:
        return points[0][1]
    if minutes >= points[-1][0]:
        return points[-1][1]
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t0 <= minutes <= t1:
            frac = (minutes - t0) / (t1 - t0)
            return v0 + frac * (v1 - v0)
    return None


def _tide_trend_at(curve: list[dict], minutes: float, window_minutes: float = 60) -> float | None:
    """潮位の変化率(cm/時)。プラスなら上げ潮、マイナスなら下げ潮で、流れの強さの目安になる。"""
    before = _tide_level_at(curve, minutes - window_minutes / 2)
    after = _tide_level_at(curve, minutes + window_minutes / 2)
    if before is None or after is None:
        return None
    return (after - before) / (window_minutes / 60)


def _win_payouts_by_boat(session: Session, race_id: int) -> dict[int, int]:
    rows = session.exec(
        select(Payout).where(Payout.race_id == race_id, Payout.bet_type == "win")
    ).all()
    result: dict[int, int] = {}
    for row in rows:
        try:
            boat = int(row.combination)
        except ValueError:
            continue
        result[boat] = row.amount
    return result


def build_race_features(session: Session, race_id: int) -> pd.DataFrame:
    """1レース分を6行(艇ごと)のDataFrameにする。preview/resultが無ければその列はNaN。"""
    race = session.get(Race, race_id)
    if race is None:
        raise ValueError(f"race_id={race_id} が見つかりません")

    racer_entries = {
        e.racer_boat_number: e
        for e in session.exec(
            select(RacerEntry).where(RacerEntry.race_id == race_id)
        ).all()
    }
    preview_weather = session.exec(
        select(PreviewWeather).where(PreviewWeather.race_id == race_id)
    ).first()
    preview_entries = {
        e.racer_boat_number: e
        for e in session.exec(
            select(PreviewEntry).where(PreviewEntry.race_id == race_id)
        ).all()
    }
    result_weather = session.exec(
        select(RaceWeather).where(RaceWeather.race_id == race_id)
    ).first()
    result_entries = {
        e.racer_boat_number: e
        for e in session.exec(
            select(ResultEntry).where(ResultEntry.race_id == race_id)
        ).all()
    }
    win_payouts = _win_payouts_by_boat(session, race_id)

    tide_daily = session.exec(
        select(TideDaily).where(
            TideDaily.date == race.date, TideDaily.stadium_number == race.stadium_number
        )
    ).first()
    tide_level_cm = None
    tide_trend_cm_per_hour = None
    tide_name = tide_daily.tide_name if tide_daily else None
    moon_age = tide_daily.moon_age if tide_daily else None
    if tide_daily and race.closed_at:
        curve = json.loads(tide_daily.curve_json)
        closed_at = datetime.strptime(race.closed_at, "%Y-%m-%d %H:%M:%S")
        minutes = closed_at.hour * 60 + closed_at.minute
        tide_level_cm = _tide_level_at(curve, minutes)
        tide_trend_cm_per_hour = _tide_trend_at(curve, minutes)

    # 展示タイム・展示ST・全国勝率は艇間の相対順位が効くので先に集計しておく
    exhibition_times = {
        b: e.exhibition_time
        for b, e in preview_entries.items()
        if e.exhibition_time is not None
    }
    exhibition_ranks = _rank(exhibition_times, ascending=True)  # 速いほど1位

    preview_starts = {
        b: e.start_timing for b, e in preview_entries.items() if e.start_timing is not None
    }
    preview_start_ranks = _rank(preview_starts, ascending=True)  # 早い(小さい)ほど1位

    rows = []
    for boat_number in range(1, 7):
        racer = racer_entries.get(boat_number)
        preview_entry = preview_entries.get(boat_number)
        result_entry = result_entries.get(boat_number)

        place = result_entry.racer_place_number if result_entry else None
        is_normal = place in _NORMAL_PLACES

        rows.append(
            {
                "race_id": race_id,
                "date": race.date,
                "stadium_number": race.stadium_number,
                "water_type": STADIUM_WATER_TYPE.get(race.stadium_number),
                "race_number": race.number,
                "racer_boat_number": boat_number,
                "course_number": (
                    preview_entry.course_number if preview_entry else boat_number
                ),
                "racer_number": racer.racer_number if racer else None,
                "racer_name": racer.racer_name if racer else None,
                "class_number": racer.racer_class_number if racer else None,
                "age": racer.racer_age if racer else None,
                "weight": racer.racer_weight if racer else None,
                "flying_count": racer.racer_flying_count if racer else None,
                "late_count": racer.racer_late_count if racer else None,
                "average_start_timing": (
                    racer.racer_average_start_timing if racer else None
                ),
                "national_win_rate": (
                    racer.racer_national_top_1_percent if racer else None
                ),
                "national_top2_rate": (
                    racer.racer_national_top_2_percent if racer else None
                ),
                "national_top3_rate": (
                    racer.racer_national_top_3_percent if racer else None
                ),
                "local_win_rate": racer.racer_local_top_1_percent if racer else None,
                "local_top2_rate": racer.racer_local_top_2_percent if racer else None,
                "local_top3_rate": racer.racer_local_top_3_percent if racer else None,
                "motor_top2_rate": (
                    racer.racer_assigned_motor_top_2_percent if racer else None
                ),
                "motor_top3_rate": (
                    racer.racer_assigned_motor_top_3_percent if racer else None
                ),
                "boat_top2_rate": (
                    racer.racer_assigned_boat_top_2_percent if racer else None
                ),
                "boat_top3_rate": (
                    racer.racer_assigned_boat_top_3_percent if racer else None
                ),
                # 直前情報(preview): レース前の実測値
                "exhibition_time": (
                    preview_entry.exhibition_time if preview_entry else None
                ),
                "exhibition_time_rank": exhibition_ranks.get(boat_number),
                "preview_start_timing": (
                    preview_entry.start_timing if preview_entry else None
                ),
                "preview_start_timing_rank": preview_start_ranks.get(boat_number),
                "tilt_adjustment": (
                    preview_entry.tilt_adjustment if preview_entry else None
                ),
                "weight_adjustment": (
                    preview_entry.weight_adjustment if preview_entry else None
                ),
                "wind_speed": preview_weather.wind_speed if preview_weather else None,
                "wind_direction_number": (
                    preview_weather.wind_direction_number if preview_weather else None
                ),
                "wave_height": preview_weather.wave_height if preview_weather else None,
                "weather_number": (
                    preview_weather.weather_number if preview_weather else None
                ),
                "air_temperature": (
                    preview_weather.air_temperature if preview_weather else None
                ),
                "water_temperature": (
                    preview_weather.water_temperature if preview_weather else None
                ),
                # 潮汐(天文推算値、レース発走時刻に線形補間)
                "tide_level_cm": tide_level_cm,
                "tide_trend_cm_per_hour": tide_trend_cm_per_hour,
                "tide_name": tide_name,
                "moon_age": moon_age,
                # 結果(target)
                "place_number": place,
                "is_normal_finish": is_normal,
                "is_win": bool(is_normal and place == 1),
                "is_top2": bool(is_normal and place is not None and place <= 2),
                "is_top3": bool(is_normal and place is not None and place <= 3),
                "win_payout": win_payouts.get(boat_number),
                # 参考: 結果側の気象実測値(バックテスト専用、直前情報とはズレうる)
                "result_wind_speed": result_weather.wind_speed if result_weather else None,
                "result_wave_height": result_weather.wave_height if result_weather else None,
            }
        )

    return pd.DataFrame(rows)


def _rank(values: dict[int, float], ascending: bool) -> dict[int, int]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda kv: kv[1], reverse=not ascending)
    return {boat: i + 1 for i, (boat, _) in enumerate(ordered)}


def build_dataset(
    session: Session,
    stadium_numbers: list[int] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    require_result: bool = True,
) -> pd.DataFrame:
    """条件に合うレースをまとめてfeature DataFrameにする(全レース分を縦に結合)。"""
    query = select(Race.id)
    if stadium_numbers is not None:
        query = query.where(Race.stadium_number.in_(stadium_numbers))
    if date_from is not None:
        query = query.where(Race.date >= date_from)
    if date_to is not None:
        query = query.where(Race.date <= date_to)

    race_ids = session.exec(query).all()
    frames = []
    for race_id in race_ids:
        df = build_race_features(session, race_id)
        if require_result and df["place_number"].isna().all():
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
