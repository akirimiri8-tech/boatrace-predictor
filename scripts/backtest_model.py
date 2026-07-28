"""ロジスティック回帰モデルを学習し、時系列分割でベースラインと比較する。

学習期間より後のデータだけでテストすることで、未来のデータを
学習に混ぜてしまう(過学習・リーク)のを防いでいる。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlmodel import Session

from boatrace_predictor.backtest.evaluate import (
    evaluate_course_baseline,
    evaluate_model_picks,
    time_based_split,
)
from boatrace_predictor.config import STADIUMS, settings
from boatrace_predictor.features.dataset import build_dataset
from boatrace_predictor.models.scoring import predict_win_probability, train
from boatrace_predictor.persistence.db import engine


def main() -> None:
    with Session(engine) as session:
        df = build_dataset(session, stadium_numbers=settings.target_stadiums)

    df = df[df["is_normal_finish"]].reset_index(drop=True)
    train_df, test_df = time_based_split(df, train_frac=0.7)

    train_dates = (train_df["date"].min(), train_df["date"].max())
    test_dates = (test_df["date"].min(), test_df["date"].max())
    print(f"学習期間: {train_dates[0]} 〜 {train_dates[1]} ({train_df['race_id'].nunique()}レース)")
    print(f"検証期間: {test_dates[0]} 〜 {test_dates[1]} ({test_df['race_id'].nunique()}レース)")
    print()

    model = train(train_df)
    test_probs = predict_win_probability(model, test_df)

    model_result = evaluate_model_picks(test_df, test_probs, label="ロジスティック回帰モデル")
    baseline_result = evaluate_course_baseline(test_df, course_number=1, label="1コース固定")

    for r in (baseline_result, model_result):
        print(f"--- {r.label} ---")
        print(f"  対象レース数: {r.n_races}")
        print(f"  的中率: {r.hit_rate:.1%}")
        print(f"  単勝回収率: {r.roi:.1%}")
        print()

    print("会場別(検証期間):")
    for stadium in settings.target_stadiums:
        sub = test_df[test_df["stadium_number"] == stadium]
        sub_probs = test_probs[sub.index]
        m = evaluate_model_picks(sub, sub_probs, label="モデル")
        b = evaluate_course_baseline(sub, course_number=1, label="1コース固定")
        name = STADIUMS[stadium]
        print(f"  {stadium}={name}: モデル的中率{m.hit_rate:.1%}/回収率{m.roi:.1%}"
              f" vs 1コース的中率{b.hit_rate:.1%}/回収率{b.roi:.1%}")


if __name__ == "__main__":
    main()
