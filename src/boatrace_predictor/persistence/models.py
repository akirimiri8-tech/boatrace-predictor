"""SQLite永続化スキーマ (SQLModel)。

Race は (date, stadium_number, number) で一意。
1レースに対して RacerEntry(出走表)、PreviewWeather/PreviewEntry(直前情報)、
RaceWeather/ResultEntry/Payout(結果) がそれぞれ別タイミングで埋まる
(出走表は開催前、直前情報は発走少し前、結果は終了後)。
"""

from sqlmodel import Field, SQLModel


class Race(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    date: str = Field(index=True)  # YYYY-MM-DD
    stadium_number: int = Field(index=True)
    number: int  # 1R〜12R (APIのrace_number)
    closed_at: str | None = None
    day_number: int | None = None  # 開催日次(最終日など)
    grade_number: int | None = None
    title: str | None = None
    subtitle: str | None = None
    distance: int | None = None

    __table_args__ = ({"sqlite_autoincrement": True},)


class RacerEntry(SQLModel, table=True):
    """出走表(racers)の1艇分。"""

    id: int | None = Field(default=None, primary_key=True)
    race_id: int = Field(foreign_key="race.id", index=True)

    racer_boat_number: int
    racer_name: str | None = None
    racer_number: int | None = None
    racer_class_number: int | None = None
    racer_branch_number: int | None = None
    racer_birthplace_number: int | None = None
    racer_age: int | None = None
    racer_weight: float | None = None
    racer_flying_count: int | None = None
    racer_late_count: int | None = None
    racer_average_start_timing: float | None = None
    racer_national_top_1_percent: float | None = None
    racer_national_top_2_percent: float | None = None
    racer_national_top_3_percent: float | None = None
    racer_local_top_1_percent: float | None = None
    racer_local_top_2_percent: float | None = None
    racer_local_top_3_percent: float | None = None
    racer_assigned_motor_number: int | None = None
    racer_assigned_motor_top_2_percent: float | None = None
    racer_assigned_motor_top_3_percent: float | None = None
    racer_assigned_boat_number: int | None = None
    racer_assigned_boat_top_2_percent: float | None = None
    racer_assigned_boat_top_3_percent: float | None = None


class PreviewWeather(SQLModel, table=True):
    """直前情報(preview)の気象・水面情報(レース単位、1件)。発走少し前の実測値。"""

    id: int | None = Field(default=None, primary_key=True)
    race_id: int = Field(foreign_key="race.id", index=True, unique=True)

    wind_speed: int | None = None
    wind_direction_number: int | None = None
    wave_height: int | None = None
    weather_number: int | None = None
    air_temperature: float | None = None
    water_temperature: float | None = None


class PreviewEntry(SQLModel, table=True):
    """直前情報(preview.racers)の1艇分。展示航走の実測値。"""

    id: int | None = Field(default=None, primary_key=True)
    race_id: int = Field(foreign_key="race.id", index=True)

    racer_boat_number: int  # entry_number(元の艇番)
    course_number: int | None = None  # 直前情報時点の実際の進入コース
    start_timing: float | None = None  # 展示スタートタイミング
    weight: float | None = None
    weight_adjustment: float | None = None
    exhibition_time: float | None = None
    tilt_adjustment: float | None = None


class RaceWeather(SQLModel, table=True):
    """result の気象・水面情報(レース単位、1件)。レース後の実測値。"""

    id: int | None = Field(default=None, primary_key=True)
    race_id: int = Field(foreign_key="race.id", index=True, unique=True)

    wind_speed: int | None = None
    wind_direction_number: int | None = None
    wave_height: int | None = None
    weather_number: int | None = None
    air_temperature: float | None = None
    water_temperature: float | None = None
    technique_number: int | None = None


class ResultEntry(SQLModel, table=True):
    """result の1艇分の着順情報。"""

    id: int | None = Field(default=None, primary_key=True)
    race_id: int = Field(foreign_key="race.id", index=True)

    racer_boat_number: int
    racer_course_number: int | None = None
    racer_start_timing: float | None = None
    racer_place_number: int | None = None
    racer_number: int | None = None
    racer_name: str | None = None


class TideDaily(SQLModel, table=True):
    """1日1会場分の潮汐データ。20分おきの潮位カーブはJSON文字列で持つ
    (レース時刻の潮位・変化率は特徴量生成時に補間して計算する)。"""

    id: int | None = Field(default=None, primary_key=True)
    date: str = Field(index=True)  # YYYY-MM-DD
    stadium_number: int = Field(index=True)

    tide_name: str | None = None  # 潮名: 大潮/中潮/小潮/長潮/若潮
    moon_age: float | None = None
    curve_json: str  # [{"time": "00:00", "cm": 131.9}, ...] のJSON文字列

    __table_args__ = ({"sqlite_autoincrement": True},)


class Payout(SQLModel, table=True):
    """舟券種別ごとの払戻金。bet_type: trifecta/trio/exacta/quinella/quinella_place/win/place"""

    id: int | None = Field(default=None, primary_key=True)
    race_id: int = Field(foreign_key="race.id", index=True)

    bet_type: str
    combination: str
    amount: int


class Prediction(SQLModel, table=True):
    """日次運用ループでの予想ログ。結果が出る前(発走前)に保存し、後から
    ResultEntry/Payoutと突き合わせて的中率・回収率を評価する。
    同じレース・同じモデルで複数回予想した場合は最新のものだけ残す。"""

    id: int | None = Field(default=None, primary_key=True)
    race_id: int = Field(foreign_key="race.id", index=True)
    predicted_at: str  # ISO8601タイムスタンプ(いつ予想したか)
    model_name: str  # 例: "logistic_regression"

    predicted_1st: int
    predicted_2nd: int | None = None
    predicted_3rd: int | None = None

    __table_args__ = ({"sqlite_autoincrement": True},)


class PredictionScore(SQLModel, table=True):
    """予想時点での艇ごとのスコア(1着確率など)。"""

    id: int | None = Field(default=None, primary_key=True)
    prediction_id: int = Field(foreign_key="prediction.id", index=True)

    racer_boat_number: int
    score: float
