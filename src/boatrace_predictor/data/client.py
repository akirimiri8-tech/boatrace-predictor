"""Boatrace Open API(統合api/v1)クライアント。

https://github.com/BoatraceOpenAPI/api

非公式API。データが存在しない日付(2026-01-01より前、または未来すぎる日)は
404を返すので、その場合は「その日はデータなし」として空リストを返す。

1日1回のリクエストで全会場・全レースの出走表+直前情報+結果が返る
(会場ごとに分かれていた旧v3 programs/resultsとは異なる)。
"""

from datetime import date

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from boatrace_predictor.config import settings
from boatrace_predictor.data.schemas import ApiV1Response, RaceProgram

_RETRYABLE = (httpx.TransportError, httpx.HTTPStatusError)


class BoatraceOpenAPIClient:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            base_url=settings.api_base_url,
            timeout=settings.request_timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BoatraceOpenAPIClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _get_json(self, path: str) -> dict | None:
        url = f"/{path}"
        resp = self._client.get(url)
        if resp.status_code == 404:
            logger.debug(f"データなし: {url}")
            return None
        resp.raise_for_status()
        return resp.json()

    def fetch_day(
        self, target_date: date, stadiums: set[int] | None = None
    ) -> list[RaceProgram]:
        """指定日の全レースを取得する。stadiumsを指定すればその場コードのみに絞る。"""
        path = f"api/{settings.api_version}/{target_date.year}/{target_date:%Y%m%d}.json"
        raw = self._get_json(path)
        if raw is None:
            return []

        parsed = ApiV1Response.model_validate(raw)
        races: list[RaceProgram] = []
        for stadium in parsed.programs.stadiums.values():
            for race in stadium.races.values():
                if stadiums is None or race.stadium_number in stadiums:
                    races.append(race)
        return races

    def fetch_today(self, stadiums: set[int] | None = None) -> list[RaceProgram]:
        raw = self._get_json(f"api/{settings.api_version}/today.json")
        if raw is None:
            return []
        parsed = ApiV1Response.model_validate(raw)
        races = []
        for stadium in parsed.programs.stadiums.values():
            for race in stadium.races.values():
                if stadiums is None or race.stadium_number in stadiums:
                    races.append(race)
        return races
