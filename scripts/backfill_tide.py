"""対象会場の潮汐データ(潮汐736 API)を取得してDBに保存する。

天文計算による推算値なので、レース結果と違って未来の日付も取得できる。

使い方:
    python scripts/backfill_tide.py --start 2026-01-01 --end 2026-07-27
"""

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger
from sqlmodel import Session

from boatrace_predictor.config import STADIUM_TIDE_STATIONS, settings
from boatrace_predictor.data.tide_client import TideClient
from boatrace_predictor.persistence.db import engine, init_db, save_tide_day


def daterange(start: date, end: date):
    for n in range((end - start).days + 1):
        yield start + timedelta(days=n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--stadiums", type=int, nargs="*", default=settings.target_stadiums
    )
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    init_db()

    total = 0
    with TideClient() as client, Session(engine) as session:
        for stadium in args.stadiums:
            if stadium not in STADIUM_TIDE_STATIONS:
                logger.warning(f"stadium={stadium} は潮汐観測地点が未設定のためスキップ")
                continue
            pc, hc = STADIUM_TIDE_STATIONS[stadium]
            for d in daterange(start, end):
                chart = client.fetch_day(pc, hc, d)
                if chart is not None:
                    save_tide_day(session, d.isoformat(), stadium, chart)
                    total += 1
                time.sleep(args.sleep)
            session.commit()
            logger.info(f"stadium={stadium}: 完了")

    logger.info(f"完了: {total}件(会場x日)を保存")


if __name__ == "__main__":
    main()
