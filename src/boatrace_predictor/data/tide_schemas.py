"""潮汐736 API (https://tide736.net/) のレスポンススキーマ。

実データ(2026-07-20分)を取得してフィールド名を確認した上で定義している。
潮位は天文計算による推算値なので、過去・未来どちらの日付でも取得できる。
"""

from pydantic import BaseModel


class TidePoint(BaseModel):
    time: str
    cm: float


class MoonInfo(BaseModel):
    brightness: str | None = None
    age: float | None = None
    title: str | None = None  # 潮名: 大潮/中潮/小潮/長潮/若潮


class DayChart(BaseModel):
    moon: MoonInfo
    tide: list[TidePoint]  # 20分おきの潮位(cm)
    flood: list[TidePoint] = []  # 満潮
    edd: list[TidePoint] = []  # 干潮


class TideBody(BaseModel):
    chart: dict[str, DayChart]  # キーは"YYYY-MM-DD"


class TideResponse(BaseModel):
    status: int
    message: str | None = None
    tide: TideBody

    def day(self) -> DayChart:
        return next(iter(self.tide.chart.values()))
