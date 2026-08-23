"""Boatrace Open API (統合api/v1) のレスポンススキーマ。

https://boatraceopenapi.github.io/api/v1/{YYYY}/{YYYYMMDD}.json
実データ(2026-07-20分)を取得してフィールド名を確認した上で定義している。
2026-01-01以降のデータに対応(それ以前は404)。

racers(出走表)/preview(直前情報)/result(結果) がレース単位で1つに
まとまっているのが旧v3(programs/results別リポジトリ)との違い。
preview には展示航走時の実測風速・波高・展示タイム・チルト調整が入っており、
これが「レース前の予想」に使える気象/直前情報。
"""

from pydantic import BaseModel


class Racer(BaseModel):
    """出走表の選手データ(racers[N])。"""

    entry_number: int
    # 出走取消等で枠だけ存在し選手が未確定の場合、identity情報がnullになることがある
    name: str | None = None
    number: int | None = None
    rank_number: int | None = None
    branch_number: int | None = None
    birthplace_number: int | None = None
    age: int | None = None
    weight: float | None = None
    flying_count: int | None = None
    late_count: int | None = None
    average_start_timing: float | None = None
    national_win_rate: float | None = None
    national_top_2_percent: float | None = None
    national_top_3_percent: float | None = None
    local_win_rate: float | None = None
    local_top_2_percent: float | None = None
    local_top_3_percent: float | None = None
    motor_number: int | None = None
    motor_top_2_percent: float | None = None
    motor_top_3_percent: float | None = None
    boat_number: int | None = None
    boat_top_2_percent: float | None = None
    boat_top_3_percent: float | None = None


class PreviewRacer(BaseModel):
    """直前情報の選手データ(preview.racers[N])。展示航走の実測値。"""

    entry_number: int
    course_number: int | None = None  # 直前情報時点の実際の進入コース
    start_timing: float | None = None  # 展示スタートタイミング
    weight: float | None = None
    weight_adjustment: float | None = None
    exhibition_time: float | None = None
    tilt_adjustment: float | None = None


class Preview(BaseModel):
    """直前情報(preview)。レース発走の少し前に実測値で埋まる。"""

    date: str
    stadium_number: int
    race_number: int
    wind_speed: int | None = None
    wind_direction_number: int | None = None
    wave_height: int | None = None
    weather_number: int | None = None
    air_temperature: float | None = None
    water_temperature: float | None = None
    racers: dict[str, PreviewRacer] = {}


class ResultRacer(BaseModel):
    entry_number: int
    course_number: int | None = None
    start_timing: float | None = None
    place_number: int | None = None
    # 欠場等でレース自体に出走しなかった場合、選手識別情報がnullになることがある
    number: int | None = None
    name: str | None = None


class PayoutEntry(BaseModel):
    # 稀にcombination/amountがnullになるケースがある(2026-08-22、複勝等の払戻で確認)。
    # 原因不明(該当着順が存在しないレースなど)だが欠損として扱い、保存時にスキップする。
    combination: str | None = None
    amount: int | None = None


class Payouts(BaseModel):
    trifecta: list[PayoutEntry] = []
    trio: list[PayoutEntry] = []
    exacta: list[PayoutEntry] = []
    quinella: list[PayoutEntry] = []
    quinella_place: list[PayoutEntry] = []
    win: list[PayoutEntry] = []
    place: list[PayoutEntry] = []


class RaceResult(BaseModel):
    date: str
    stadium_number: int
    race_number: int
    wind_speed: int | None = None
    wind_direction_number: int | None = None
    wave_height: int | None = None
    weather_number: int | None = None
    air_temperature: float | None = None
    water_temperature: float | None = None
    technique_number: int | None = None
    racers: dict[str, ResultRacer] = {}
    payouts: Payouts | None = None


class RaceProgram(BaseModel):
    """1レース分の全情報(出走表+直前情報+結果)。"""

    date: str
    stadium_number: int
    race_number: int
    closed_at: str | None = None
    grade_number: int | None = None
    title: str | None = None
    subtitle: str | None = None
    distance: int | None = None
    day_number: int | None = None
    racers: dict[str, Racer] = {}
    preview: Preview | None = None
    result: RaceResult | None = None


class Stadium(BaseModel):
    races: dict[str, RaceProgram] = {}


class ProgramsRoot(BaseModel):
    stadiums: dict[str, Stadium] = {}


class ApiV1Response(BaseModel):
    programs: ProgramsRoot
