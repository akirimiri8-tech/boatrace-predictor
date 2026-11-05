import pandas as pd

from boatrace_predictor.features.dataset import _add_race_relative_features


def _race() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "national_win_rate": [4.0, 7.0, 6.0, 5.0, 5.0, 3.0],
            "class_number": [3, 1, 1, 2, 3, 3],
            "motor_top2_rate": [30.0, 50.0, 40.0, 20.0, 35.0, 10.0],
        }
    )


def test_relative_features_for_weak_boat_1_with_strong_rivals() -> None:
    df = _add_race_relative_features(_race())
    boat1 = df.iloc[0]
    assert boat1["national_win_rate_rank"] == 5
    assert boat1["national_win_rate_gap_to_best_other"] == -3.0  # 4.0 - 最高の7.0
    assert boat1["n_a1_others"] == 2


def test_best_boat_gap_is_measured_against_second_best() -> None:
    df = _add_race_relative_features(_race())
    best = df.iloc[1]
    assert best["national_win_rate_rank"] == 1
    assert best["national_win_rate_gap_to_best_other"] == 1.0  # 7.0 - 次点の6.0
    assert best["n_a1_others"] == 1  # 自分自身はA1でも数えない


def test_motor_rank_is_descending() -> None:
    df = _add_race_relative_features(_race())
    assert df["motor_top2_rate_rank"].tolist() == [4, 1, 2, 5, 3, 6]
