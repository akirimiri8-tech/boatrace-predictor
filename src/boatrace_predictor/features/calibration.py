"""コース別に勝率予測のキャリブレーション(予測確率のズレ)を補正する。

2026-08-22に発覚した問題: ロジスティック回帰の予測確率は全体平均では
実際の勝率とほぼ一致する(58.0%予測 vs 56.3%実績)が、コース別に見ると
系統的にズレている。特に2〜5コースは予測確率30〜50%帯で実際の勝率を
大きく下回っていた(例: コース4は予測28.4% vs 実際21.2%)。
1コースは逆に過小評価気味(予測40.3% vs 実際45.0%)。

このズレは単純な「確率が全体的に高い/低い」ではなく「同じ予測確率でも
コースによって実際の勝率が違う」という構造なので、確率値そのものだけを
入力にした通常のキャリブレーション(1本のisotonic回帰)では直せない。
コースごとに別々の補正曲線を作る必要がある。

2026-09-15、学習期間とは別に確保した検証専用データ(calib_df)でこの
コース別isotonic回帰を検証: 補正後はコース1が46.1%(実際45.0%)、
コース4が21.2%(実際21.2%、完全一致)などと大幅に改善した
(train/calib/testの3分割、calibで補正曲線をフィットしtestで検証済み)。
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

_MIN_SAMPLES_PER_COURSE = 30  # これ未満なら補正せず生の予測確率をそのまま使う


@dataclass
class CourseCalibration:
    calibrators: dict[int, IsotonicRegression] = field(default_factory=dict)


def fit_course_calibration(
    calib_df: pd.DataFrame, predicted_prob: pd.Series
) -> CourseCalibration:
    """calib_df(学習には使っていないレース)と、そのレースに対する予測確率から
    コースごとのisotonic回帰(予測確率 -> 実際の勝率)をフィットする。"""
    frame = calib_df.assign(_pred_prob=predicted_prob.values)
    calibrators: dict[int, IsotonicRegression] = {}
    for course, group in frame.groupby("course_number"):
        if len(group) < _MIN_SAMPLES_PER_COURSE:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(group["_pred_prob"], group["is_win"].astype(int))
        calibrators[int(course)] = iso
    return CourseCalibration(calibrators=calibrators)


def apply_course_calibration(
    df: pd.DataFrame, predicted_prob: pd.Series, calibration: CourseCalibration
) -> pd.Series:
    """コース別の補正曲線を適用する。該当コースの補正曲線が無い(サンプル不足)場合は
    生の予測確率をそのまま返す。"""
    result = predicted_prob.copy()
    for course, iso in calibration.calibrators.items():
        mask = (df["course_number"] == course).to_numpy()
        if not mask.any():
            continue
        idx = predicted_prob.index[mask]
        result.loc[idx] = iso.predict(predicted_prob.loc[idx].to_numpy())
    return result
