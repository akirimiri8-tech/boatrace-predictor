"""ベースライン精度の確認。

一番単純な「1コースが1着になる」という競艇の定説だけでどこまで当たるか、
会場ごとの的中率・回収率(単勝100円あたり)を算出する。
これがモデルの最低ライン(超えるべき基準)になる。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sqlmodel import Session

from boatrace_predictor.config import STADIUMS, settings
from boatrace_predictor.features.dataset import build_dataset
from boatrace_predictor.persistence.db import engine


def summarize(df: pd.DataFrame, label: str) -> None:
    normal = df[df["is_normal_finish"]]
    n_races = normal["race_id"].nunique()

    course1 = normal[normal["course_number"] == 1]
    hit_rate = course1["is_win"].mean()

    # 単勝100円を全レースでコース1に賭けたと仮定した場合の回収率
    stake = len(course1) * 100
    payout = course1.loc[course1["is_win"], "win_payout"].fillna(0).sum()
    roi = payout / stake if stake else float("nan")

    print(f"--- {label} ---")
    print(f"  正常終了レース数: {n_races}")
    print(f"  1コース1着率: {hit_rate:.1%}")
    print(f"  1コース単勝回収率(100円/レース均等): {roi:.1%}")


def main() -> None:
    with Session(engine) as session:
        for stadium in settings.target_stadiums:
            df = build_dataset(session, stadium_numbers=[stadium])
            summarize(df, f"{stadium}={STADIUMS[stadium]}")

        df_all = build_dataset(session, stadium_numbers=settings.target_stadiums)
        summarize(df_all, "全対象会場")


if __name__ == "__main__":
    main()
