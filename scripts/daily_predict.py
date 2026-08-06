"""当日(または指定日)の出走表・直前情報を取得し、モデルで予想してDBにログ保存する。

実際の舟券は買わない(ペーパー予想運用、CLAUDE.mdフェーズ5)。予想対象日より
前の結果データだけでモデルを学習するので、未来のデータを混ぜていない。
翌日以降に scripts/daily_report.py で結果と突き合わせて精度を確認する。

発走時刻(closed_at)を過ぎたレースはデフォルトでスキップする。直前情報は
API側で発走後にも更新されることがあり、発走済みレースに対して予想を作ると
「本当にレース前に分かっていた情報」ではなくなってしまうため
(2026-08-03、7Rの予想が再現できない問題が発生して判明)。
過去日を指定して研究目的で無理やり予想を作りたい場合は --allow-finished を使う。

このスクリプトは1日1回ではなく、レース開催中は30分〜1時間おきに繰り返し
実行する運用を想定している(1回で全レース分の直前情報が揃うわけではないため)。
既に予想済みのレースは再度上書きされる(直前情報が更新され次第、最新の内容で
上書きする設計)。

使い方:
    python scripts/daily_predict.py
    python scripts/daily_predict.py --date 2026-07-29 --allow-finished
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
from boatrace_predictor.models import ranking
from boatrace_predictor.models.scoring import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    predict_win_probability,
    train,
)
from boatrace_predictor.persistence.db import engine, init_db, save_prediction, save_race

# 2026-08-06: LightGBMランキングモデルに切り替え(単勝/複勝の回収率で
# ロジスティック回帰を上回るようになったため)。ロジスティック回帰は
# 比較用に引き続きログし続ける。PRIMARY_MODELの予想だけを標準出力に表示する。
PRIMARY_MODEL = "lightgbm_ranking"


def _fit_models(history):
    """(model_name, predict_fn) のリストを返す。predict_fn(df) -> pd.Series[win_prob的なスコア]"""
    logistic_model = train(history)
    preprocessor, ranker = ranking.train_ranker(history)
    return [
        (PRIMARY_MODEL, lambda df: ranking.predict_scores(preprocessor, ranker, df)),
        ("logistic_regression", lambda df: predict_win_probability(logistic_model, df)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD(未指定なら本日)")
    parser.add_argument("--stadiums", type=int, nargs="*", default=settings.target_stadiums)
    parser.add_argument(
        "--allow-finished",
        action="store_true",
        help="発走済みレースも予想対象にする(研究目的の過去日再現用。通常は使わない)",
    )
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
        models = _fit_models(history)
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

        # 3. レースごとに予想してログ保存(発走済みは原則スキップ)
        now = datetime.now()
        predicted_at = now.isoformat(timespec="seconds")
        n_predicted = 0
        n_skipped_finished = 0
        for race in races:
            if race.closed_at and not args.allow_finished:
                closed_at = datetime.strptime(race.closed_at, "%Y-%m-%d %H:%M:%S")
                if closed_at <= now:
                    n_skipped_finished += 1
                    continue

            race_id = race_ids[(race.stadium_number, race.race_number)]
            df = build_race_features(session, race_id)
            if df["national_win_rate"].isna().all():
                continue  # 出走表自体が無い(会場コードの取り違え等)場合はスキップ
            df = apply_course_encodings(df, encodings)
            features_snapshot = df[
                ["racer_boat_number"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
            ].to_dict("records")

            primary_ranked = None
            primary_scores = None
            for model_name, predict_fn in models:
                scores_series = predict_fn(df)
                ranked = (
                    df.assign(_score=scores_series)
                    .sort_values("_score", ascending=False)["racer_boat_number"]
                    .tolist()
                )
                scores = dict(zip(df["racer_boat_number"], scores_series))
                save_prediction(
                    session, race_id, predicted_at, model_name, ranked, scores, features_snapshot
                )
                if model_name == PRIMARY_MODEL:
                    primary_ranked, primary_scores = ranked, scores
            n_predicted += 1

            name = STADIUMS[race.stadium_number]
            # LightGBMランキングモデルのスコアは確率ではない(0〜1に収まらない生スコア)ので
            # %表示はしない。ロジスティック回帰のような確率が欲しい場合は
            # daily_report.py --model logistic_regression 側のログを見る
            print(
                f"{race.date} {name} {race.race_number}R: "
                f"予想 {primary_ranked[0]}-{primary_ranked[1]}-{primary_ranked[2]} "
                f"(1着スコア: {primary_scores[primary_ranked[0]]:.2f})"
            )
        session.commit()

        logger.info(
            f"完了: {n_predicted}レース分の予想を保存"
            f"(発走済みでスキップ: {n_skipped_finished}件)"
        )


if __name__ == "__main__":
    main()
