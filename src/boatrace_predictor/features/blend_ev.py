"""モデルの勝率と市場の2連単オッズを混ぜた確率から、期待値の高い2連単の並びを探す(2026-11-10)。

過去約10万レースの検証(確定オッズで選んだ場合)で『期待値が高い並びほど回収率が高い』傾向が出た手法。
log p_blend = A * log p_model + (1-A) * log q_market (A=0.2、検証の前半で決めた値)、
期待値 = p_blend x オッズ。p_modelはPlackett-Luceで勝率から2連単化、q_marketはオッズの逆数の正規化。
"""

import itertools

import numpy as np

A = 0.2
PAIRS = list(itertools.permutations(range(1, 7), 2))
NAMES = [f"{a}-{b}" for a, b in PAIRS]


def blended_exacta_picks(
    win_probs: dict[int, float], exacta_odds: dict[str, float], min_ev: float = 1.1
) -> list[tuple[str, float, float, float]]:
    """(並び, オッズ, 混合確率, 期待値) のリストを返す。入力が不完全(6艇揃わない/30通り揃わない/
    1.0倍以下を含む)なら空。"""
    if len(win_probs) != 6 or any(i not in win_probs for i in range(1, 7)):
        return []
    if len(exacta_odds) < 30 or any(exacta_odds.get(n, 0.0) <= 1.0 for n in NAMES):
        return []
    p = np.array([max(win_probs[i], 1e-6) for i in range(1, 7)])
    p /= p.sum()
    od = np.array([exacta_odds[n] for n in NAMES])
    pm = np.array([p[a - 1] * p[b - 1] / (1 - p[a - 1]) for a, b in PAIRS])
    pm /= pm.sum()
    q = 1 / od
    q /= q.sum()
    pb = np.exp(A * np.log(pm + 1e-12) + (1 - A) * np.log(q))
    pb /= pb.sum()
    ev = pb * od
    return [(NAMES[j], float(od[j]), float(pb[j]), float(ev[j])) for j in range(30) if ev[j] >= min_ev]
