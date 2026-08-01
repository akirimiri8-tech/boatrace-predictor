"""当日(または指定日)の出走表・直前情報を取得し、モデルで予想してDBにログ保存する。

実際の舟券は買わない(ペーパー予想運用、CLAUDE.mdフェーズ5)。予想対象日より
前の結果データだけでモデルを学習するので、未来のデータを混ぜていない。
翌日以降に scripts/daily_report.py で結果と突き合わせて精度を確認する。

使い方:
    python scripts/daily_predict.py
    python scripts/daily_predict.py --date 2026-07-29
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger
from sqlmodel import Session

from boatrace_predictor.config import STADIUMS, settings
from boatrace_predictor.data.client import BoatraceOpenAPIClient
from boatrace_predictor.features.dataset import build_dataset, build_race_features
from boatrace_predictor.features.encodings import apply_course_encodings, fit_course_encodings
from boatrace_predictor.models.scoring import predict_win_probability, train
from boatrace_predictor.persistence.db import engine, init_db, save_prediction, save_race

MODEL_NAME = "logistic_regression"  # 単勝/複勝が主指標という方針上、こちらを正式運用に採用


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD(未指定なら本日)")
    parser.add_argument("--stadiums", type=int, nargs="*", default=settings.target_stadiums)
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    stadiums = set(args.stadiums)

    init_db()

    with Session(engine) as session:
        # 1. target_dateより前の結果データだけで学習する(未来データを混ぜない)
        history = build_dataset(
            session,
            stadium_numbers=args.stadiums,
            date_to=(target_date - timedelta(days=1)).isoformat(),
        )
        history = history[history["is_normal_finish"]].reset_index(drop=True)
        if history.empty or history["is_win"].sum() == 0:
            logger.error(f"{target_date}より前の学習データが無いため予想できません")
            return

        encodings = fit_course_encodings(history)
        history = apply_course_encodings(history, encodings)
        model = train(history)
        logger.info(
            f"学習データ: {history['race_id'].nunique()}レース"
            f"({history['date'].min()}〜{history['date'].max()})"
        )

        # 2. 当日データ取得・保存(出走表+直前情報。結果はまだ無い想定)
        with BoatraceOpenAPIClient() as client:
            races = client.fetch_day(target_date, stadiums=stadiums)
        if not races:
            logger.warning(f"{target_date}: 対象会場のレースが見つかりませんでした")
            return

        race_ids = {}
        for race in races:
            race_id = save_race(session, race)
            race_ids[(race.stadium_number, race.race_number)] = race_id
        session.commit()

        # 3. レースごとに予想してログ保存
        predicted_at = datetime.now().isoformat(timespec="seconds")
        n_predicted = 0
        for race in races:
            race_id = race_ids[(race.stadium_number, race.race_number)]
            df = build_race_features(session, race_id)
            if df["national_win_rate"].isna().all():
                continue  # 出走表自体が無い(会場コードの取り違え等)場合はスキップ
            df = apply_course_encodings(df, encodings)
            probs = predict_win_probability(model, df)

            ranked = (
                df.assign(_score=probs)
                .sort_values("_score", ascending=False)["racer_boat_number"]
                .tolist()
            )
            scores = dict(zip(df["racer_boat_number"], probs))
            save_prediction(session, race_id, predicted_at, MODEL_NAME, ranked, scores)
            n_predicted += 1

            name = STADIUMS[race.stadium_number]
            print(
                f"{race.date} {name} {race.race_number}R: "
                f"予想 {ranked[0]}-{ranked[1]}-{ranked[2]} "
                f"(1着確率: {scores[ranked[0]]:.1%})"
            )
        session.commit()

        logger.info(f"完了: {n_predicted}レース分の予想を保存")


if __name__ == "__main__":
    main()
