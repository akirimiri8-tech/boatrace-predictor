import pandas as pd

from boatrace_predictor.features.calibration import apply_course_calibration, fit_course_calibration


def _calib_df() -> pd.DataFrame:
    # コース1: 予測確率0.4のレースが40件、実際は32件勝っている(予測は過小評価)
    # コース2: 予測確率0.4のレースが40件、実際は8件しか勝っていない(予測は過大評価)
    rows = []
    for i in range(40):
        rows.append({"course_number": 1, "is_win": i < 32})
        rows.append({"course_number": 2, "is_win": i < 8})
    return pd.DataFrame(rows)


def test_fit_course_calibration_creates_calibrator_per_course_with_enough_samples() -> None:
    calib_df = _calib_df()
    predicted_prob = pd.Series([0.4] * len(calib_df), index=calib_df.index)

    calibration = fit_course_calibration(calib_df, predicted_prob)

    assert 1 in calibration.calibrators
    assert 2 in calibration.calibrators


def test_apply_course_calibration_corrects_course_specific_bias() -> None:
    calib_df = _calib_df()
    predicted_prob = pd.Series([0.4] * len(calib_df), index=calib_df.index)
    calibration = fit_course_calibration(calib_df, predicted_prob)

    df = pd.DataFrame(
        [
            {"course_number": 1},
            {"course_number": 2},
        ]
    )
    raw_prob = pd.Series([0.4, 0.4], index=df.index)

    corrected = apply_course_calibration(df, raw_prob, calibration)

    # コース1は実際8/10=0.8勝っていたので補正後は上振れするはず
    assert corrected.iloc[0] > 0.4
    # コース2は実際2/10=0.2しか勝っていないので補正後は下振れするはず
    assert corrected.iloc[1] < 0.4


def test_apply_course_calibration_falls_back_to_raw_for_unseen_course() -> None:
    calib_df = _calib_df()
    predicted_prob = pd.Series([0.4] * len(calib_df), index=calib_df.index)
    calibration = fit_course_calibration(calib_df, predicted_prob)

    df = pd.DataFrame([{"course_number": 6}])
    raw_prob = pd.Series([0.35], index=df.index)

    corrected = apply_course_calibration(df, raw_prob, calibration)

    # コース6の補正曲線は無い(サンプル不足)ので、生の予測確率のまま
    assert corrected.iloc[0] == 0.35


def test_fit_course_calibration_skips_courses_with_too_few_samples() -> None:
    calib_df = pd.DataFrame(
        [{"course_number": 3, "is_win": True} for _ in range(5)]
    )
    predicted_prob = pd.Series([0.5] * len(calib_df), index=calib_df.index)

    calibration = fit_course_calibration(calib_df, predicted_prob)

    assert 3 not in calibration.calibrators
