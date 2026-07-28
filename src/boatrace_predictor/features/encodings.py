"""会場×コース、選手×コースの過去勝率を特徴量にする(自前データのみ、外部APIなし)。

学習期間(train_df)からだけ統計を作り、検証期間(test_df)にも同じ統計を
適用する。学習データに存在しない組み合わせ(初出走の選手・場など)は
上位の統計にフォールバックする(選手×コース → 会場×コース → 全体平均)。

選手×コースはレース数が少ない選手も多いため、会場×コースの勝率を
事前分布とした縮小推定(shrinkage)にしている。prior_weightは
「会場×コースの勝率をどれだけ信じるか」を仮想レース数で表したもの。

検証時の注記: racer_course_win_rate は prior_weight=10 だと単勝的中率が
54.1%→50.5%に悪化した(データが約5か月分・2会場分と少なく、選手×コースの
サンプルが薄すぎてノイズを拾ったため)。prior_weight=100前後まで上げると
ベースライン付近まで戻るが、それはほぼ venue_course_win_rate と同じ値に
収束しているだけで実質的な追加情報がない。そのためモデル側
(models/scoring.py の NUMERIC_FEATURES)では現状 venue_course_win_rate
のみを使い、racer_course_win_rate は未採用にしている。データが増えたら
再評価する。
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CourseEncodings:
    venue_course_rate: dict[tuple[int, int], float] = field(default_factory=dict)
    racer_course_wins: dict[tuple[int, int], float] = field(default_factory=dict)
    racer_course_n: dict[tuple[int, int], float] = field(default_factory=dict)
    global_mean: float = 1 / 6
    prior_weight: float = 10.0


def fit_course_encodings(train_df: pd.DataFrame, prior_weight: float = 10.0) -> CourseEncodings:
    global_mean = train_df["is_win"].mean() if len(train_df) else 1 / 6

    venue_course = train_df.groupby(["stadium_number", "course_number"])["is_win"].mean()
    racer_course = train_df.groupby(["racer_number", "course_number"])["is_win"].agg(
        ["sum", "count"]
    )

    return CourseEncodings(
        venue_course_rate=venue_course.to_dict(),
        racer_course_wins=racer_course["sum"].to_dict(),
        racer_course_n=racer_course["count"].to_dict(),
        global_mean=global_mean,
        prior_weight=prior_weight,
    )


def apply_course_encodings(df: pd.DataFrame, encodings: CourseEncodings) -> pd.DataFrame:
    df = df.copy()

    venue_course_keys = list(zip(df["stadium_number"], df["course_number"]))
    df["venue_course_win_rate"] = [
        encodings.venue_course_rate.get(k, encodings.global_mean) for k in venue_course_keys
    ]

    racer_course_keys = list(zip(df["racer_number"], df["course_number"]))
    wins = pd.Series(
        [encodings.racer_course_wins.get(k, 0.0) for k in racer_course_keys], index=df.index
    )
    n = pd.Series(
        [encodings.racer_course_n.get(k, 0.0) for k in racer_course_keys], index=df.index
    )
    prior = df["venue_course_win_rate"]
    df["racer_course_win_rate"] = (wins + encodings.prior_weight * prior) / (
        n + encodings.prior_weight
    )

    return df
