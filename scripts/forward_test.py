"""購入時点(締切12/7/2分前)の2連単オッズで選んだ場合の「先読み検証」(お金は賭けない)。

過去の大量検証(確定オッズで選んでいる)は、締切前に実際に買えるオッズでは成り立たない
可能性がある。ExactaOddsSnapshot に溜めた『買える時点のオッズ』で、
  混ぜた確率 = モデル20% + 市場80% (対数で混合)、期待値 = 混ぜた確率 x その時点のオッズ
が閾値以上の並びを買ったことにし、実際の払戻(確定オッズ)で損益を出す。
モデル確率は本番のlogistic_regression(コース別補正済み)の勝率をPlackett-Luceで2連単化したもの。

使い方: python scripts/forward_test.py [DBパス] [--stake 5000]
"""

import itertools
import sqlite3
import sys

import numpy as np

A = 0.2  # モデルの重み(過去検証の前半で決めた値)
THRESHOLDS = (1.0, 1.1, 1.2, 1.3)
PAIRS = list(itertools.permutations(range(1, 7), 2))
NAMES = [f"{a}-{b}" for a, b in PAIRS]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    db = args[0] if args else "boatrace.db"
    stake = 5000
    if "--stake" in sys.argv:
        stake = int(sys.argv[sys.argv.index("--stake") + 1])
    con = sqlite3.connect(db)

    # 各レースの最新のlogistic予想(勝率スコア)
    pred = {}
    for rid, pid in con.execute(
        "SELECT race_id, MAX(id) FROM prediction WHERE model_name='logistic_regression' GROUP BY race_id"
    ):
        sc = dict(con.execute("SELECT racer_boat_number, score FROM predictionscore WHERE prediction_id=?", (pid,)))
        if len(sc) == 6:
            v = np.array([max(sc[i], 1e-6) for i in range(1, 7)])
            pred[rid] = v / v.sum()

    pay = {}
    for rid, comb, amt in con.execute("SELECT race_id, combination, amount FROM payout WHERE bet_type='exacta'"):
        pay.setdefault(rid, {})[comb] = amt / 100

    snaps = {}
    for rid, mb, comb, od in con.execute("SELECT race_id, minutes_before, combination, odds FROM exactaoddssnapshot"):
        snaps.setdefault((rid, mb), {})[comb] = od

    print(f"予想あり{len(pred)}レース / 結果あり{len(pay)}レース / スナップショット{len(snaps)}件")
    for mb in (12, 7, 2):
        print(f"\n=== 締切{mb}分前のオッズで選ぶ(賭け金1点{stake:,}円) ===")
        data = []
        for (rid, m), o in snaps.items():
            if m != mb or rid not in pred or rid not in pay or len(o) < 30 or any(v <= 1.0 for v in o.values()):
                continue
            if len(pay[rid]) != 1:
                continue
            od = np.array([o[n] for n in NAMES])
            p = pred[rid]
            pm = np.array([p[a - 1] * p[b - 1] / (1 - p[a - 1]) for a, b in PAIRS])
            pm /= pm.sum()
            q = 1 / od
            q /= q.sum()
            pb = np.exp(A * np.log(pm + 1e-12) + (1 - A) * np.log(q))
            pb /= pb.sum()
            data.append((od, pb, pay[rid]))
        print(f"対象レース: {len(data)}")
        if not data:
            continue
        for th in THRESHOLDS:
            n_bet = n_hit = 0
            ret = 0.0
            races = set()
            for i, (od, pb, py) in enumerate(data):
                for j, nm in enumerate(NAMES):
                    if pb[j] * od[j] >= th:
                        n_bet += 1
                        races.add(i)
                        if nm in py:
                            n_hit += 1
                            ret += py[nm]
            if n_bet == 0:
                print(f"  期待値{th}以上: 買い目なし")
                continue
            print(
                f"  期待値{th}以上: {n_bet}点/{len(races)}レース 的中{n_hit} "
                f"回収率{ret / n_bet:.0%} 損益{(ret - n_bet) * stake:+,.0f}円"
            )


if __name__ == "__main__":
    main()
