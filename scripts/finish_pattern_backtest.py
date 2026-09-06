"""features/finish_pattern.py (出目パターンによる2着・3着の組み直し)が
実際に券種別回収率を改善するかを検証する。

比較対象:
- 現状のdaily_predict.pyと同じ「スコア順に1〜3着を並べるだけ」
- 1着はスコア最大のまま、2着・3着だけ出目パターン(1着コースから見た
  2着コースの経験確率など)で選び直したもの

同じモデルスコア・同じ検証データを使い、並べ方の違いだけを比較する
(モデル自体の優劣ではなく、出目パターンによる並べ替えの効果を見たいため)。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlmodel import Session

from boatrace_predictor.backtest.bet_types import BET_TYPE_LABELS, evaluate_bet_types_from_order
from boatrace_predictor.backtest.evaluate import time_based_split
from boatrace_predictor.config import settings
from boatrace_predictor.features.dataset import build_dataset
from boatrace_predictor.features.encodings import apply_course_encodings, fit_course_encodings
from boatrace_predictor.features.finish_pattern import fit_finish_pattern, rerank_with_pattern
from boatrace_predictor.models import ranking
from boatrace_predictor.persistence.db import engine


def _naive_order(df, scores) -> dict[int, list[int]]:
    scored = df.assign(_score=scores)
    order = {}
    for race_id, group in scored.groupby("race_id"):
        order[race_id] = group.sort_values("_score", ascending=False)["racer_boat_number"].tolist()
    return order


def print_bet_table(session: Session, order: dict[int, list[int]], label: str) -> None:
    results = evaluate_bet_types_from_order(session, order)
    print(f"=== {label} ===")
    print(f"{'券種':6s} {'的中率':>8s} {'回収率':>8s} {'的中時平均払戻':>12s}")
    for r in sorted(results, key=lambda r: r.roi, reverse=True):
        name = BET_TYPE_LABELS[r.bet_type]
        print(f"{name:6s} {r.hit_rate:8.1%} {r.roi:8.1%} {r.avg_payout_when_hit:12,.0f}円")
    print()


def main() -> None:
    with Session(engine) as session:
        df = build_dataset(session, stadium_numbers=settings.target_stadiums)
        df = df[df["is_normal_finish"]].reset_index(drop=True)
        train_df, test_df = time_based_split(df, train_frac=0.7)

        encodings = fit_course_encodings(train_df)
        train_df = apply_course_encodings(train_df, encodings)
        test_df = apply_course_encodings(test_df, encodings)

        pattern = fit_finish_pattern(train_df)

        print(f"検証レース数: {test_df['race_id'].nunique()}\n")

        preprocessor, ranker = ranking.train_ranker(train_df)
        scores = ranking.predict_scores(preprocessor, ranker, test_df)

        naive = _naive_order(test_df, scores)
        reranked = rerank_with_pattern(test_df, scores, pattern)
        reranked_with_start = rerank_with_pattern(test_df, scores, pattern, use_start_rank=True)

        # 1着の予想がどれだけ変わっていないか確認(意図通り、1着は変えていないはず)
        same_first = sum(
            1 for rid in naive if naive[rid][:1] == reranked[rid][:1]
        )
        print(f"1着予想が一致: {same_first}/{len(naive)}レース\n")

        print_bet_table(session, naive, "現状(スコア順)")
        print_bet_table(session, reranked, "出目パターン(コースのみ)で2着・3着を組み直し")
        print_bet_table(
            session,
            reranked_with_start,
            "出目パターン(コース+1着艇のスタート順位)で2着・3着を組み直し",
        )


if __name__ == "__main__":
    main()
