"""ログした予想(Predictionテーブル)を実際の結果と突き合わせて的中率・回収率を集計する。

daily_predict.py で保存した予想のうち、結果が確定しているレースだけを対象にする。
単勝・複勝を主指標にする方針(CLAUDE.md参照、2026-07-28決定)なので、
多艇券種は参考値として表示するが的中数が少ないうちは判断材料にしない。

使い方:
    python scripts/daily_report.py
    python scripts/daily_report.py --start 2026-07-01 --end 2026-07-31
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlmodel import Session, select

from boatrace_predictor.backtest.bet_types import BET_TYPE_LABELS, evaluate_bet_types_from_order
from boatrace_predictor.persistence.db import engine
from boatrace_predictor.persistence.models import Payout, Prediction, Race

_PRIMARY_BET_TYPES = {"win", "place"}  # 主指標。他は参考値。


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    parser.add_argument("--model", default="logistic_regression")
    args = parser.parse_args()

    with Session(engine) as session:
        query = select(Prediction, Race).join(Race, Prediction.race_id == Race.id).where(
            Prediction.model_name == args.model
        )
        if args.start:
            query = query.where(Race.date >= args.start)
        if args.end:
            query = query.where(Race.date <= args.end)
        rows = session.exec(query).all()

        if not rows:
            print("該当する予想ログがありません")
            return

        race_ids_with_result = set(
            session.exec(
                select(Payout.race_id).where(
                    Payout.race_id.in_([p.race_id for p, r in rows])
                )
            ).all()
        )

        order: dict[int, list[int]] = {}
        n_pending = 0
        for prediction, race in rows:
            if prediction.race_id not in race_ids_with_result:
                n_pending += 1
                continue
            picks = [prediction.predicted_1st, prediction.predicted_2nd, prediction.predicted_3rd]
            order[prediction.race_id] = [b for b in picks if b is not None]

        if n_pending:
            print(f"(結果未確定のため除外: {n_pending}件)")
        if not order:
            print("結果が確定した予想がまだありません")
            return

        results = evaluate_bet_types_from_order(session, order)

    print(f"評価対象: {len(order)}レース(モデル: {args.model})\n")
    print(f"{'券種':6s} {'的中率':>8s} {'回収率':>8s} {'的中時平均払戻':>12s}")
    for r in sorted(results, key=lambda r: r.bet_type not in _PRIMARY_BET_TYPES):
        label = BET_TYPE_LABELS[r.bet_type]
        tag = "" if r.bet_type in _PRIMARY_BET_TYPES else "(参考)"
        print(f"{label:6s} {r.hit_rate:8.1%} {r.roi:8.1%} {r.avg_payout_when_hit:12,.0f}円 {tag}")


if __name__ == "__main__":
    main()
