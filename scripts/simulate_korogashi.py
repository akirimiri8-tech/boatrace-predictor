"""「ころがし」(勝った分を次のレースにそのまま賭ける複利式)のシミュレーション。

通常の評価(daily_report.py)は毎回100円均等ベットで的中率・回収率を見るが、
ころがしは1回外れると資金が尽きる(オールイン方式)。日をまたいで転がし続けるのは
非現実的なので、日ごとにリセットして「その日、何レース目まで転がせたか」
「最終的にいくらになったか」を計算する。

使い方:
    python scripts/simulate_korogashi.py --start 2026-08-01 --end 2026-08-22
    python scripts/simulate_korogashi.py --start 2026-08-01 --end 2026-08-22 --bet-type place --seed-yen 1000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlmodel import Session, select

from boatrace_predictor.backtest.bet_types import BET_TYPE_LABELS
from boatrace_predictor.persistence.db import engine
from boatrace_predictor.persistence.models import Payout, Prediction, Race

_BET_TYPE_SIZE = {"win": 1, "place": 1}  # ころがしは単勝/複勝のみ対応(1艇の的中判定のみ)


def _is_hit(bet_type: str, boat: int, combination: str) -> bool:
    return combination == str(boat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--model", default="lightgbm_ranking")
    parser.add_argument("--bet-type", default="win", choices=["win", "place"])
    parser.add_argument("--seed-yen", type=int, default=1000, help="1日の元手(円)")
    parser.add_argument(
        "--target-multiplier",
        type=float,
        default=None,
        help="元手の何倍に達したら打ち止めにするか(例: 2で2倍達成時点で利確して終了)",
    )
    args = parser.parse_args()

    with Session(engine) as session:
        rows = session.exec(
            select(Prediction, Race)
            .join(Race, Prediction.race_id == Race.id)
            .where(
                Prediction.model_name == args.model,
                Race.date >= args.start,
                Race.date <= args.end,
            )
            .order_by(Race.date, Race.stadium_number, Race.number)
        ).all()

        if not rows:
            print("該当する予想ログがありません")
            return

        # 日付ごとにグループ化(会場・レース番号順=発走順に近い並び)
        by_date: dict[str, list[tuple[Prediction, Race]]] = {}
        for prediction, race in rows:
            by_date.setdefault(race.date, []).append((prediction, race))

        label = BET_TYPE_LABELS[args.bet_type]
        print(f"券種: {label} / 1日の元手: {args.seed_yen}円 / モデル: {args.model}\n")

        total_days = 0
        survived_full_day = 0
        target_hit_days = 0
        busted_days = 0
        final_balances = []
        target_balance = args.seed_yen * args.target_multiplier if args.target_multiplier else None

        for day, day_rows in sorted(by_date.items()):
            balance = args.seed_yen
            n_races_rolled = 0
            busted = False
            target_hit = False

            for prediction, race in day_rows:
                payouts = session.exec(
                    select(Payout).where(
                        Payout.race_id == prediction.race_id, Payout.bet_type == args.bet_type
                    )
                ).all()
                if not payouts:
                    continue  # 結果未確定(まだレースが終わっていない)のでスキップ

                pick = prediction.predicted_1st
                hit = next((p for p in payouts if _is_hit(args.bet_type, pick, p.combination)), None)
                n_races_rolled += 1

                if hit is not None:
                    balance = balance * (hit.amount / 100)
                    if target_balance is not None and balance >= target_balance:
                        target_hit = True
                        break  # 目標額に到達したので打ち止め(利確)
                else:
                    balance = 0
                    busted = True
                    break

            total_days += 1
            final_balances.append(balance)
            if not busted:
                survived_full_day += 1
            if target_hit:
                target_hit_days += 1
            if busted:
                busted_days += 1

            if target_hit:
                status = f"目標達成({args.target_multiplier}倍)で打ち止め"
            elif busted:
                status = f"{n_races_rolled}レース目で終了(全損)"
            else:
                status = "その日の全レース終了(目標未達)"
            print(f"{day}: {n_races_rolled}レース挑戦 → {status}, 最終資金 {balance:,.0f}円")

        print()
        avg_final = sum(final_balances) / len(final_balances)
        print(f"対象日数: {total_days}日")
        if target_balance is not None:
            print(f"目標({args.target_multiplier}倍={target_balance:,.0f}円)達成した日: "
                  f"{target_hit_days}日 ({target_hit_days/total_days:.0%})")
        print(f"全損した日: {busted_days}日 ({busted_days/total_days:.0%})")
        print(f"平均最終資金: {avg_final:,.0f}円 (元手{args.seed_yen}円)")
        print(f"日次平均リターン: {avg_final/args.seed_yen:.1%}")


if __name__ == "__main__":
    main()
