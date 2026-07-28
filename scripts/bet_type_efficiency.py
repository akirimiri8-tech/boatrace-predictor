"""券種ごとの回収率比較。学習期間より後のデータだけで評価する。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlmodel import Session

from boatrace_predictor.backtest.bet_types import BET_TYPE_LABELS, evaluate_bet_types
from boatrace_predictor.backtest.evaluate import time_based_split
from boatrace_predictor.config import settings
from boatrace_predictor.features.dataset import build_dataset
from boatrace_predictor.models.scoring import predict_win_probability, train
from boatrace_predictor.persistence.db import engine


def main() -> None:
    with Session(engine) as session:
        df = build_dataset(session, stadium_numbers=settings.target_stadiums)
        df = df[df["is_normal_finish"]].reset_index(drop=True)
        train_df, test_df = time_based_split(df, train_frac=0.7)

        model = train(train_df)
        test_probs = predict_win_probability(model, test_df)

        results = evaluate_bet_types(session, test_df, test_probs)

    print(f"検証レース数: {test_df['race_id'].nunique()}")
    print(f"{'券種':6s} {'的中率':>8s} {'回収率':>8s} {'的中時平均払戻':>12s}")
    for r in sorted(results, key=lambda r: r.roi, reverse=True):
        label = BET_TYPE_LABELS[r.bet_type]
        print(
            f"{label:6s} {r.hit_rate:8.1%} {r.roi:8.1%} {r.avg_payout_when_hit:12,.0f}円"
        )


if __name__ == "__main__":
    main()
