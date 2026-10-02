"""ケリー基準の「先読みログ」(KellyLogテーブル)を実際の結果と突き合わせて検証する。

2026-09-30: 過去データでの一回のバックテスト(学習に使っていないtest分割、オッズ付き
1734レース)でケリー基準による賭け方を試したところ単勝ROI183%という好結果が出た。
ただし同じ枠組みの別のバックテスト(2026-09-16、EV>=1.2閾値)では逆に的中率12.2%・
ROI60.9%という結果で、これと矛盾する。過去の検証でも「一回のバックテストで良さそうに
見えた施策が、大きいデータで再検証すると崩れた」ことが何度もあった(例: 低自信時の
1号艇フォールバック)ため、実際にお金を賭ける前に daily_predict.py で日々ログを
溜めるだけにして(KellyLog、お金は賭けない)、このスクリプトで後から答え合わせする。

このレポートが示す「ケリー基準でプラス判定」件数・的中率・ROIが、数週間分のログが
溜まった時点でもバックテストの数字(的中率28.4%、ROI183%前後)に近ければ、
実運用に進めるかどうかの判断材料にする。大きく外れていたら(特に過去のEV>=1.2検証
のように悪い数字に戻っていたら)、やはりまだ使えないという結論になる。

使い方:
    python scripts/kelly_report.py
    python scripts/kelly_report.py --start 2026-09-30 --end 2026-10-15
    python scripts/kelly_report.py --daily-budget 100000 --kelly-scale 0.25
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlmodel import Session, select

from boatrace_predictor.persistence.db import engine
from boatrace_predictor.persistence.models import KellyLog, Payout, Race, ResultEntry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--daily-budget", type=float, default=100_000, help="1日あたりの予算(円)"
    )
    parser.add_argument(
        "--kelly-scale", type=float, default=1.0, help="フルケリーに対する倍率(0.25なら1/4ケリー)"
    )
    args = parser.parse_args()

    with Session(engine) as session:
        query = select(KellyLog, Race).join(Race, KellyLog.race_id == Race.id)
        if args.start:
            query = query.where(Race.date >= args.start)
        if args.end:
            query = query.where(Race.date <= args.end)
        rows = session.exec(query).all()

        # 同じレース・同じ艇で複数回ログしている場合は最新(predicted_at最大)だけ使う
        latest: dict[tuple[int, int], tuple[KellyLog, Race]] = {}
        for log, race in rows:
            key = (log.race_id, log.racer_boat_number)
            if key not in latest or log.predicted_at > latest[key][0].predicted_at:
                latest[key] = (log, race)

        race_ids = {log.race_id for log, _ in latest.values()}
        results = session.exec(
            select(ResultEntry).where(ResultEntry.race_id.in_(race_ids))
        ).all()
        winner_by_race: dict[int, int | None] = {}
        for r in results:
            if r.racer_place_number == 1:
                winner_by_race[r.race_id] = r.racer_boat_number

        payouts = session.exec(
            select(Payout).where(Payout.race_id.in_(race_ids)).where(Payout.bet_type == "win")
        ).all()
        payout_by_race: dict[int, dict[int, int]] = {}
        for p in payouts:
            payout_by_race.setdefault(p.race_id, {})[int(p.combination)] = p.amount

        n_unresolved = 0
        picks = []  # (date, race_id, boat, prob, odds, kelly_f, is_win)
        for (race_id, boat), (log, race) in latest.items():
            if race_id not in winner_by_race:
                n_unresolved += 1
                continue
            is_win = winner_by_race[race_id] == boat
            picks.append((race.date, race_id, boat, log.calibrated_prob, log.win_odds, log.kelly_fraction, is_win))

        print(f"(結果未確定のため除外: {n_unresolved}件)")
        print(f"ログ総数: {len(picks)}件(艇ごと)")

        positive = [p for p in picks if p[5] > 0]
        if not positive:
            print("ケリー基準でプラス判定のログがまだありません")
            return

        n_hit = sum(1 for p in positive if p[6])
        stake = 100
        payout_sum = sum(p[4] * stake for p in positive if p[6])
        cost = len(positive) * stake
        print(f"\n=== ケリー基準でプラス判定: {len(positive)}件(均等{stake}円ベット換算) ===")
        print(f"  的中: {n_hit}件 ({n_hit / len(positive):.1%})")
        print(f"  ROI: {payout_sum / cost:.1%}")

        # 日予算内に正規化したシミュレーション(scripts/daily_report.pyと違い、ケリー係数で
        # 賭け金を傾斜配分する。1日の合計がdaily-budgetを超える場合は比例配分で収める)
        by_date: dict[str, list] = {}
        for p in positive:
            by_date.setdefault(p[0], []).append(p)

        cum_profit = 0.0
        total_staked = 0.0
        worst_day = 0.0
        n_bet = 0
        n_bet_hit = 0
        for d, bets in sorted(by_date.items()):
            scaled = [(p, p[5] * args.kelly_scale) for p in bets]
            total_f = sum(f for _, f in scaled)
            if total_f > 1.0:
                scaled = [(p, f / total_f) for p, f in scaled]
            day_profit = 0.0
            for p, f in scaled:
                stake_amt = args.daily_budget * f
                total_staked += stake_amt
                n_bet += 1
                if p[6]:
                    n_bet_hit += 1
                    day_profit += stake_amt * (p[4] - 1)
                else:
                    day_profit -= stake_amt
            cum_profit += day_profit
            worst_day = min(worst_day, day_profit)

        print(
            f"\n=== 日予算{args.daily_budget:,.0f}円・ケリー倍率{args.kelly_scale}でのシミュレーション ==="
        )
        print(f"  対象日数: {len(by_date)}日 / 賭けた回数: {n_bet}件(的中{n_bet_hit}件)")
        print(f"  総投資額: {total_staked:,.0f}円")
        print(f"  累計損益: {cum_profit:+,.0f}円")
        if total_staked > 0:
            print(f"  ROI: {(cum_profit + total_staked) / total_staked:.1%}")
        print(f"  最悪の1日: {worst_day:+,.0f}円")


if __name__ == "__main__":
    main()
