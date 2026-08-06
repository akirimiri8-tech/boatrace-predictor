import numpy as np
import pandas as pd

from boatrace_predictor.models.scoring import predict_win_probability, train


def _synthetic_dataset(n_races: int = 200, seed: int = 0) -> pd.DataFrame:
    """コース1が勝ちやすいという分かりやすい構造を持つ合成データ。"""
    rng = np.random.default_rng(seed)
    rows = []
    for race_id in range(n_races):
        # コース番号が小さいほど勝率が高くなるように重み付けして1着を決める
        win_weights = np.array([6, 3, 2, 1.5, 1, 0.8])
        win_boat = rng.choice(np.arange(1, 7), p=win_weights / win_weights.sum())
        for boat in range(1, 7):
            rows.append(
                {
                    "race_id": race_id,
                    "date": f"2026-01-{(race_id % 28) + 1:02d}",
                    "stadium_number": 24,
                    "race_number": 1,
                    "racer_boat_number": boat,
                    "course_number": boat,
                    "class_number": rng.choice([1, 2, 3, 4]),
                    "water_type": rng.choice(["海水", "汽水", "淡水"]),
                    "age": rng.integers(20, 60),
                    "weight": rng.uniform(45, 60),
                    "flying_count": 0,
                    "late_count": 0,
                    "average_start_timing": rng.uniform(0.1, 0.2),
                    "national_win_rate": rng.uniform(3, 7),
                    "national_top2_rate": rng.uniform(20, 50),
                    "national_top3_rate": rng.uniform(30, 60),
                    "local_win_rate": rng.uniform(3, 7),
                    "local_top2_rate": rng.uniform(20, 50),
                    "local_top3_rate": rng.uniform(30, 60),
                    "motor_top2_rate": rng.uniform(20, 50),
                    "motor_top3_rate": rng.uniform(30, 60),
                    "boat_top2_rate": rng.uniform(20, 50),
                    "boat_top3_rate": rng.uniform(30, 60),
                    "exhibition_time": rng.uniform(6.5, 7.0),
                    "exhibition_time_rank": boat,
                    "wake_adjusted_exhibition_time": rng.uniform(-0.2, 0.2),
                    "meet_trend_avg_place": rng.uniform(1, 6),
                    "preview_start_timing": rng.uniform(0.05, 0.25),
                    "preview_start_timing_rank": boat,
                    "tilt_adjustment": rng.choice([-0.5, 0, 0.5]),
                    "weight_adjustment": rng.uniform(0, 2),
                    "wind_speed": rng.integers(0, 8),
                    "wave_height": rng.integers(0, 5),
                    "weather_number": 1,
                    "wind_direction_number": 1,
                    "air_temperature": rng.uniform(15, 30),
                    "water_temperature": rng.uniform(15, 28),
                    "tide_level_cm": rng.uniform(50, 200),
                    "tide_trend_cm_per_hour": rng.uniform(-30, 30),
                    "tide_name": rng.choice(["大潮", "中潮", "小潮", "長潮", "若潮"]),
                    "moon_age": rng.uniform(0, 29),
                    "venue_course_win_rate": rng.uniform(0.1, 0.5),
                    "racer_course_win_rate": rng.uniform(0.1, 0.5),
                    "is_win": boat == win_boat,
                    "win_payout": 250 if boat == win_boat else None,
                }
            )
    return pd.DataFrame(rows)


def test_train_and_predict_returns_valid_probabilities() -> None:
    df = _synthetic_dataset()
    model = train(df)
    probs = predict_win_probability(model, df)

    assert len(probs) == len(df)
    assert probs.between(0, 1).all()


def test_model_learns_course_advantage() -> None:
    df = _synthetic_dataset()
    model = train(df)
    probs = predict_win_probability(model, df)

    avg_prob_by_course = df.assign(p=probs).groupby("course_number")["p"].mean()
    # 合成データはコース1が最も勝ちやすく作っているので、学習後もコース1の平均予測確率が最大のはず
    assert avg_prob_by_course.idxmax() == 1
