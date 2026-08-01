"""券種(単勝・複勝・2連単・2連複・拡連複・3連単・3連複)ごとの回収率比較。

モデルのスコアが高い順に「予想順位」を作り、その予想を券種ごとの
買い方に変換して実際の払戻金と突き合わせる。全券種、1レースあたり
100円均等ベットで統一して比較する(条件を揃えないと回収率が比較できない)。
"""

from dataclasses import dataclass

import pandas as pd
from sqlmodel import Session, select

from boatrace_predictor.persistence.models import Payout

_ORDERED_BET_TYPES = {"exacta", "trifecta"}
_UNORDERED_BET_TYPES = {"quinella", "quinella_place", "trio"}
_SINGLE_BET_TYPES = {"win", "place"}

BET_TYPE_LABELS = {
    "win": "単勝",
    "place": "複勝",
    "exacta": "2連単",
    "quinella": "2連複",
    "quinella_place": "拡連複",
    "trifecta": "3連単",
    "trio": "3連複",
}

BET_TYPE_SIZE = {
    "win": 1,
    "place": 1,
    "exacta": 2,
    "quinella": 2,
    "quinella_place": 2,
    "trifecta": 3,
    "trio": 3,
}


@dataclass
class BetTypeResult:
    bet_type: str
    n_races: int
    hit_rate: float
    roi: float
    avg_payout_when_hit: float


def _predicted_order(df: pd.DataFrame, predicted_prob: pd.Series) -> dict[int, list[int]]:
    scored = df.assign(_score=predicted_prob)
    order: dict[int, list[int]] = {}
    for race_id, group in scored.groupby("race_id"):
        ranked = group.sort_values("_score", ascending=False)["racer_boat_number"].tolist()
        order[race_id] = ranked
    return order


def _payouts_by_race(session: Session, race_ids: list[int]) -> dict[int, dict[str, list[tuple[str, int]]]]:
    rows = session.exec(select(Payout).where(Payout.race_id.in_(race_ids))).all()
    result: dict[int, dict[str, list[tuple[str, int]]]] = {}
    for row in rows:
        result.setdefault(row.race_id, {}).setdefault(row.bet_type, []).append(
            (row.combination, row.amount)
        )
    return result


def _is_hit(bet_type: str, pick: list[int], combination: str) -> bool:
    if bet_type in _SINGLE_BET_TYPES:
        return combination == str(pick[0])
    if bet_type in _ORDERED_BET_TYPES:
        return combination == "-".join(str(b) for b in pick)
    if bet_type in _UNORDERED_BET_TYPES:
        return set(combination.split("=")) == {str(b) for b in pick}
    raise ValueError(f"unknown bet_type: {bet_type}")


def evaluate_bet_types(
    session: Session, df: pd.DataFrame, predicted_prob: pd.Series
) -> list[BetTypeResult]:
    order = _predicted_order(df, predicted_prob)
    return evaluate_bet_types_from_order(session, order)


def evaluate_bet_types_from_order(
    session: Session, order: dict[int, list[int]]
) -> list[BetTypeResult]:
    """race_id -> 予想順位(艇番のリスト、1着予想から順)のマッピングから券種別に評価する。

    daily_predict.pyでログした実際の予想(Predictionテーブル)を評価する際にも使う。
    """
    payouts = _payouts_by_race(session, list(order.keys()))

    results = []
    for bet_type in BET_TYPE_LABELS:
        n_needed = BET_TYPE_SIZE[bet_type]
        n_races = 0
        n_hits = 0
        total_payout = 0
        for race_id, ranked in order.items():
            pick = ranked[:n_needed]
            n_races += 1
            for combination, amount in payouts.get(race_id, {}).get(bet_type, []):
                if _is_hit(bet_type, pick, combination):
                    n_hits += 1
                    total_payout += amount
                    break

        stake = n_races * 100
        roi = total_payout / stake if stake else float("nan")
        hit_rate = n_hits / n_races if n_races else float("nan")
        avg_payout = total_payout / n_hits if n_hits else float("nan")
        results.append(
            BetTypeResult(
                bet_type=bet_type,
                n_races=n_races,
                hit_rate=hit_rate,
                roi=roi,
                avg_payout_when_hit=avg_payout,
            )
        )
    return results
