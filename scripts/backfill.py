"""指定期間・対象会場のレース(出走表+直前情報+結果)を取得してDBに保存する。

使い方:
    python scripts/backfill.py --start 2026-01-01 --end 2026-07-01
    python scripts/backfill.py --start 2026-01-01 --end 2026-07-01 --stadiums 24 3

注意: 統合api/v1は2026-01-01以降のデータにのみ対応(それ以前は404で空扱いになる)。
"""

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger
from sqlmodel import Session

from boatrace_predictor.config import settings
from boatrace_predictor.data.client import BoatraceOpenAPIClient
from boatrace_predictor.persistence.db import engine, init_db, save_race


def daterange(start: date, end: date):
    for n in range((end - start).days + 1):
        yield start + timedelta(days=n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--stadiums",
        type=int,
        nargs="*",
        default=settings.target_stadiums,
        help="場コード(未指定なら config.target_stadiums)",
    )
    parser.add_argument("--sleep", type=float, default=0.3, help="APIリクエスト間隔(秒)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    stadiums = set(args.stadiums)

    init_db()

    total_races = 0
    with BoatraceOpenAPIClient() as client, Session(engine) as session:
        for d in daterange(start, end):
            races = client.fetch_day(d, stadiums=stadiums)
            for race in races:
                save_race(session, race)
                total_races += 1
            session.commit()
            logger.info(f"{d}: races={len(races)}")
            time.sleep(args.sleep)

    logger.info(f"完了: {total_races}レース分を保存")


if __name__ == "__main__":
    main()
