"""レース単位でのモデル評価。

日付で時系列分割し(未来のデータを学習に混ぜない)、各レースで
最もスコアが高い艇を「予想1着」として的中率・単勝回収率を計算する。
同じ期間・同じレース集合で「1コース固定」ベースラインとも比較する。
"""

from dataclasses import dataclass

import pandas as pd


def time_based_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].unique())
    cutoff_index = int(len(dates) * train_frac)
    cutoff_index = max(1, min(cutoff_index, len(dates) - 1))
    cutoff_date = dates[cutoff_index]
    train_df = df[df["date"] < cutoff_date].reset_index(drop=True)
    test_df = df[df["date"] >= cutoff_date].reset_index(drop=True)
    return train_df, test_df


@dataclass
class EvaluationResult:
    label: str
    n_races: int
    hit_rate: float
    roi: float


def _evaluate_picks(df: pd.DataFrame, picked_boat_by_race: pd.Series, label: str) -> EvaluationResult:
    picks = df.merge(
        picked_boat_by_race.rename("picked_boat_number"),
        left_on="race_id",
        right_index=True,
    )
    picks = picks[picks["racer_boat_number"] == picks["picked_boat_number"]]

    n_races = picks["race_id"].nunique()
    hit_rate = picks["is_win"].mean() if n_races else float("nan")
    stake = n_races * 100
    payout = picks.loc[picks["is_win"], "win_payout"].fillna(0).sum()
    roi = payout / stake if stake else float("nan")
    return EvaluationResult(label=label, n_races=n_races, hit_rate=hit_rate, roi=roi)


def evaluate_model_picks(df: pd.DataFrame, predicted_prob: pd.Series, label: str) -> EvaluationResult:
    """各レースで predicted_prob が最大の艇を予想1着として評価する。"""
    scored = df.assign(_score=predicted_prob)
    picked = scored.loc[scored.groupby("race_id")["_score"].idxmax()].set_index("race_id")[
        "racer_boat_number"
    ]
    return _evaluate_picks(df, picked, label)


def evaluate_course_baseline(df: pd.DataFrame, course_number: int, label: str) -> EvaluationResult:
    """常に指定コースを予想1着とするベースライン評価。"""
    course_rows = df[df["course_number"] == course_number]
    picked = course_rows.set_index("race_id")["racer_boat_number"]
    return _evaluate_picks(df, picked, label)
