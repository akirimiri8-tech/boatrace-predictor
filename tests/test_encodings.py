import pandas as pd

from boatrace_predictor.features.encodings import apply_course_encodings, fit_course_encodings


def _train_df() -> pd.DataFrame:
    # stadium=24, course=1: 3レース中2勝(racer_number=100が2勝、101が1勝0敗)
    rows = [
        {"stadium_number": 24, "course_number": 1, "racer_number": 100, "is_win": True},
        {"stadium_number": 24, "course_number": 1, "racer_number": 100, "is_win": True},
        {"stadium_number": 24, "course_number": 1, "racer_number": 101, "is_win": False},
        {"stadium_number": 24, "course_number": 2, "racer_number": 102, "is_win": False},
    ]
    return pd.DataFrame(rows)


def test_venue_course_win_rate_matches_empirical_mean() -> None:
    encodings = fit_course_encodings(_train_df(), prior_weight=10.0)
    assert encodings.venue_course_rate[(24, 1)] == 2 / 3
    assert encodings.venue_course_rate[(24, 2)] == 0.0


def test_racer_course_win_rate_shrinks_toward_venue_course_rate() -> None:
    encodings = fit_course_encodings(_train_df(), prior_weight=10.0)
    df = pd.DataFrame(
        [{"stadium_number": 24, "course_number": 1, "racer_number": 100}]
    )
    result = apply_course_encodings(df, encodings)

    # racer_number=100: 2勝0敗、prior=venue_course_rate(24,1)=2/3, prior_weight=10
    expected = (2 + 10 * (2 / 3)) / (2 + 10)
    assert abs(result["racer_course_win_rate"].iloc[0] - expected) < 1e-9


def test_unseen_racer_falls_back_to_venue_course_rate() -> None:
    encodings = fit_course_encodings(_train_df(), prior_weight=10.0)
    df = pd.DataFrame(
        [{"stadium_number": 24, "course_number": 1, "racer_number": 999999}]
    )
    result = apply_course_encodings(df, encodings)

    # 学習データに存在しない選手はn=0なので、racer_course_win_rate == venue_course_rateになる
    assert abs(result["racer_course_win_rate"].iloc[0] - 2 / 3) < 1e-9


def test_unseen_venue_course_falls_back_to_global_mean() -> None:
    encodings = fit_course_encodings(_train_df(), prior_weight=10.0)
    df = pd.DataFrame(
        [{"stadium_number": 999, "course_number": 9, "racer_number": 100}]
    )
    result = apply_course_encodings(df, encodings)

    global_mean = _train_df()["is_win"].mean()
    assert result["venue_course_win_rate"].iloc[0] == global_mean
