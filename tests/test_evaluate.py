import pandas as pd

from boatrace_predictor.backtest.evaluate import (
    evaluate_course_baseline,
    evaluate_model_picks,
    time_based_split,
)


def _tiny_dataset() -> pd.DataFrame:
    # 2レース、各6艇。race_id=0はcourse1が勝ち、race_id=1はcourse2が勝つ
    rows = []
    for race_id, winner_course in [(0, 1), (1, 2)]:
        for boat in range(1, 7):
            rows.append(
                {
                    "race_id": race_id,
                    "date": "2026-01-01" if race_id == 0 else "2026-01-02",
                    "racer_boat_number": boat,
                    "course_number": boat,
                    "is_win": boat == winner_course,
                    "win_payout": 300 if boat == winner_course else None,
                }
            )
    return pd.DataFrame(rows)


def test_evaluate_course_baseline_hit_rate() -> None:
    df = _tiny_dataset()
    result = evaluate_course_baseline(df, course_number=1, label="course1")
    assert result.n_races == 2
    assert result.hit_rate == 0.5  # race_id=0だけ的中
    assert result.roi == (300 / 200)


def test_evaluate_model_picks_uses_highest_score_per_race() -> None:
    df = _tiny_dataset()
    # 常に course_number をそのままスコアの逆順(艇番が小さいほど高スコア)にする
    predicted_prob = 7 - df["course_number"]
    result = evaluate_model_picks(df, predicted_prob, label="always-course1")
    # 艇番1が常に最高スコア -> course1ベースラインと同じ結果になるはず
    assert result.n_races == 2
    assert result.hit_rate == 0.5


def test_time_based_split_respects_order() -> None:
    df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]})
    train_df, test_df = time_based_split(df, train_frac=0.5)
    assert train_df["date"].max() < test_df["date"].min()
    assert len(train_df) + len(test_df) == len(df)
