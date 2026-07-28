"""潮汐736 API クライアント。

https://tide736.net/ (非公式・サポート無し・自己責任と明記されているが、
潮位は天文計算による推算値なので過去・未来どちらの日付でも安定して取得できる)
"""

from datetime import date

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from boatrace_predictor.config import settings
from boatrace_predictor.data.tide_schemas import DayChart, TideResponse

_RETRYABLE = (httpx.TransportError, httpx.HTTPStatusError)


class TideClient:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=settings.request_timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "TideClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def fetch_day(
        self, prefecture_code: int, harbor_code: int, target_date: date
    ) -> DayChart | None:
        resp = self._client.get(
            settings.tide_api_base_url,
            params={
                "pc": prefecture_code,
                "hc": harbor_code,
                "yr": target_date.year,
                "mn": target_date.month,
                "dy": target_date.day,
                "rg": "day",
            },
        )
        resp.raise_for_status()
        parsed = TideResponse.model_validate(resp.json())
        if parsed.status != 1:
            logger.warning(f"潮汐データ取得失敗: {target_date} status={parsed.status}")
            return None
        return parsed.day()
