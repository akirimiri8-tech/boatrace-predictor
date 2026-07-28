"""ロジスティック回帰 vs LightGBMランキングモデルを、単勝精度と券種別回収率の両方で比較する。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlmodel import Session

from boatrace_predictor.backtest.bet_types import BET_TYPE_LABELS, evaluate_bet_types
from boatrace_predictor.backtest.evaluate import (
    evaluate_course_baseline,
    evaluate_model_picks,
    time_based_split,
)
from boatrace_predictor.config import settings
from boatrace_predictor.features.dataset import build_dataset
from boatrace_predictor.features.encodings import apply_course_encodings, fit_course_encodings
from boatrace_predictor.models import ranking, scoring
from boatrace_predictor.persistence.db import engine


def print_bet_table(session: Session, test_df, probs, label: str) -> None:
    results = evaluate_bet_types(session, test_df, probs)
    print(f"=== {label}: 券種別 ===")
    print(f"{'券種':6s} {'的中率':>8s} {'回収率':>8s} {'的中時平均払戻':>12s}")
    for r in sorted(results, key=lambda r: r.roi, reverse=True):
        name = BET_TYPE_LABELS[r.bet_type]
        print(f"{name:6s} {r.hit_rate:8.1%} {r.roi:8.1%} {r.avg_payout_when_hit:12,.0f}円")
    print()


def main() -> None:
    with Session(engine) as session:
        df = build_dataset(session, stadium_numbers=settings.target_stadiums)
        df = df[df["is_normal_finish"]].reset_index(drop=True)
        train_df, test_df = time_based_split(df, train_frac=0.7)

        # 会場×コース・選手×コースの勝率は学習期間だけから作り、検証期間にも同じ統計を適用する
        # (テストデータの結果を学習に混ぜない)
        encodings = fit_course_encodings(train_df)
        train_df = apply_course_encodings(train_df, encodings)
        test_df = apply_course_encodings(test_df, encodings)

        print(f"検証レース数: {test_df['race_id'].nunique()}\n")

        baseline = evaluate_course_baseline(test_df, course_number=1, label="1コース固定")
        print(f"--- {baseline.label}(単勝) --- 的中率{baseline.hit_rate:.1%} 回収率{baseline.roi:.1%}\n")

        logit_model = scoring.train(train_df)
        logit_probs = scoring.predict_win_probability(logit_model, test_df)
        logit_win = evaluate_model_picks(test_df, logit_probs, label="ロジスティック回帰")
        print(f"--- {logit_win.label}(単勝) --- 的中率{logit_win.hit_rate:.1%} 回収率{logit_win.roi:.1%}\n")
        print_bet_table(session, test_df, logit_probs, "ロジスティック回帰")

        preprocessor, ranker = ranking.train_ranker(train_df)
        ranker_scores = ranking.predict_scores(preprocessor, ranker, test_df)
        ranker_win = evaluate_model_picks(test_df, ranker_scores, label="LightGBMランキング")
        print(f"--- {ranker_win.label}(単勝) --- 的中率{ranker_win.hit_rate:.1%} 回収率{ranker_win.roi:.1%}\n")
        print_bet_table(session, test_df, ranker_scores, "LightGBMランキング")


if __name__ == "__main__":
    main()
