"""1着艇のコースを起点に、2着・3着艇のコースの経験分布で予想順位を組み直す。

models/ranking.py・models/scoring.py は艇ごとのスコアを独立に出すだけで、
「1着がこのコースならこの艇が2着に来やすい」という艇同士の関係は一切
見ていない。daily_predict.py の2着・3着予想も単純にスコア順で並べているだけ。

実際のレースは決まり手(逃げ/差し/まくり/まくり差し)によって出目のパターンが
偏ることが知られている(2026-08-23、YouTube上位表示動画の調査で言及されていた内容)。
ただし動画の主張をそのまま信じるのではなく、自分たちの過去データから実際に
コース別の出目パターンを集計し、モデルのスコアと組み合わせて2着・3着予想に
使えないか検証する。

やり方: 1着のコースは既存モデルのスコアが最も高い艇をそのまま採用する
(単勝の的中率は検証済みで手を入れる理由がないため)。2着・3着だけ、
「モデルのスコア」と「1着コースから見た2着コースの経験確率(P(2着コース|1着コース))」
を掛け合わせて選び直す。3着も同様に P(3着コース|1着コース,2着コース) を使う。

学習期間(train_df)だけから統計を作り、検証期間には同じ統計を適用する
(features/encodings.py と同じ考え方)。
"""

from dataclasses import dataclass, field

import pandas as pd

_COURSES = [1, 2, 3, 4, 5, 6]
_LAPLACE_ALPHA = 1.0  # サンプルが薄い組み合わせでも確率0にならないようにする加算平滑化
_START_RANK_PRIOR_WEIGHT = 20.0  # (コース,スタート順位)条件付けの縮小推定の強さ(仮想レース数)


@dataclass
class FinishPattern:
    # (1着コース,) -> {2着コース: 確率}
    second_given_first: dict[int, dict[int, float]] = field(default_factory=dict)
    # (1着コース, 2着コース) -> {3着コース: 確率}
    third_given_first_second: dict[tuple[int, int], dict[int, float]] = field(default_factory=dict)
    # (1着コース, 1着艇がスタート順位1位だったか) -> {2着コース: 確率}
    # 2026-08-24: 1着コースだけでなく「クリーンな一番手スタートで勝ったか、
    # そうでない(まくり等の技術で勝ったか)」でも2着コースの分布が変わるはずという
    # 仮説(YouTube調査の「決まり手は風とスタートタイミングで変わる」という主張の
    # うち、風は会場ごとの向きが分からず実装できなかったのでスタート順位だけ検証)。
    # **検証の結果、course単独版との差は的中率・回収率とも±1pt未満でどちらの方向にも
    # 振れており、有意な改善とは言えなかった(train_frac 0.5〜0.8の4分割で確認、
    # scripts/finish_pattern_backtest.pyのuse_start_rank比較を参照)。不採用。**
    # rerank_with_pattern の use_start_rank はデフォルトFalseのまま、コードと
    # テストだけ残す(データが増えたら再評価の余地はある)。
    second_given_first_by_start_rank: dict[tuple[int, bool], dict[int, float]] = field(
        default_factory=dict
    )


def _race_course_order(df: pd.DataFrame) -> pd.DataFrame:
    """race_id ごとに1着・2着・3着艇のcourse_number・1着艇のスタート順位を1行にまとめる。

    ごく稀に同着(同じplace_numberの艇が複数)があり、その場合はpivotできない
    ため該当レースごと除外する(全体のごく一部なので統計への影響は無視できる)。
    """
    top3 = df[df["place_number"].isin([1, 2, 3])]
    tie_race_ids = top3.groupby(["race_id", "place_number"]).size()
    tie_race_ids = tie_race_ids[tie_race_ids > 1].index.get_level_values("race_id").unique()
    top3 = top3[~top3["race_id"].isin(tie_race_ids)]

    pivot = top3.pivot(index="race_id", columns="place_number", values="course_number")
    pivot = pivot.rename(columns={1: "course_1st", 2: "course_2nd", 3: "course_3rd"})
    pivot = pivot.dropna().astype(int)

    if "preview_start_timing_rank" in top3.columns:
        start_rank = (
            top3[top3["place_number"] == 1]
            .set_index("race_id")["preview_start_timing_rank"]
        )
        pivot = pivot.join(start_rank.rename("start_rank_1st"))
    else:
        pivot["start_rank_1st"] = pd.NA
    return pivot


def _second_given_first_counts(orders: pd.DataFrame) -> dict[int, dict[int, int]]:
    counts_by_first: dict[int, dict[int, int]] = {}
    for c1 in _COURSES:
        subset = orders[orders["course_1st"] == c1]
        counts_by_first[c1] = subset["course_2nd"].value_counts().to_dict()
    return counts_by_first


def fit_finish_pattern(train_df: pd.DataFrame) -> FinishPattern:
    orders = _race_course_order(train_df)

    raw_counts = _second_given_first_counts(orders)
    second_given_first: dict[int, dict[int, float]] = {}
    for c1 in _COURSES:
        counts = raw_counts[c1]
        candidates = [c for c in _COURSES if c != c1]
        total = sum(counts.get(c, 0) for c in candidates) + _LAPLACE_ALPHA * len(candidates)
        second_given_first[c1] = {
            c: (counts.get(c, 0) + _LAPLACE_ALPHA) / total for c in candidates
        }

    third_given_first_second: dict[tuple[int, int], dict[int, float]] = {}
    for c1 in _COURSES:
        for c2 in _COURSES:
            if c2 == c1:
                continue
            subset = orders[(orders["course_1st"] == c1) & (orders["course_2nd"] == c2)]
            counts = subset["course_3rd"].value_counts().to_dict()
            candidates = [c for c in _COURSES if c not in (c1, c2)]
            total = sum(counts.get(c, 0) for c in candidates) + _LAPLACE_ALPHA * len(candidates)
            third_given_first_second[(c1, c2)] = {
                c: (counts.get(c, 0) + _LAPLACE_ALPHA) / total for c in candidates
            }

    # (1着コース, 1着艇がスタート順位1位だったか) 条件付け。サンプルが薄いので
    # コースのみの分布(second_given_first、上で計算済み)を事前分布にした縮小推定にする
    # (features/encodings.pyのracer_course_win_rateと同じ考え方)。
    second_given_first_by_start_rank: dict[tuple[int, bool], dict[int, float]] = {}
    has_start_rank = orders["start_rank_1st"].notna().any()
    if has_start_rank:
        for c1 in _COURSES:
            candidates = [c for c in _COURSES if c != c1]
            prior = second_given_first[c1]
            for best_start in (True, False):
                subset = orders[
                    (orders["course_1st"] == c1)
                    & ((orders["start_rank_1st"] == 1) == best_start)
                ]
                counts = subset["course_2nd"].value_counts().to_dict()
                n = sum(counts.get(c, 0) for c in candidates)
                second_given_first_by_start_rank[(c1, best_start)] = {
                    c: (counts.get(c, 0) + _START_RANK_PRIOR_WEIGHT * prior[c])
                    / (n + _START_RANK_PRIOR_WEIGHT)
                    for c in candidates
                }

    return FinishPattern(
        second_given_first=second_given_first,
        third_given_first_second=third_given_first_second,
        second_given_first_by_start_rank=second_given_first_by_start_rank,
    )


def rerank_with_pattern(
    df: pd.DataFrame, scores: pd.Series, pattern: FinishPattern, use_start_rank: bool = False
) -> dict[int, list[int]]:
    """race_idごとに、1着はスコア最大の艇のまま、2着・3着だけ出目パターンで選び直す。

    use_start_rank=Trueなら、1着艇がスタート順位1位だったか(preview_start_timing_rank)も
    2着コースの確率に反映する(second_given_first_by_start_rank)。バックテストで
    course単独版との比較用に残しているオプションで、デフォルトはFalse(検証済みの
    course単独版)。

    残りの艇(4着以下相当)はスコア順のまま後ろに続ける。
    """
    scored = df.assign(_score=scores)
    order: dict[int, list[int]] = {}

    for race_id, group in scored.groupby("race_id"):
        remaining = group.sort_values("_score", ascending=False)
        boats = remaining["racer_boat_number"].tolist()
        courses = dict(zip(remaining["racer_boat_number"], remaining["course_number"]))
        score_by_boat = dict(zip(remaining["racer_boat_number"], remaining["_score"]))
        start_rank_by_boat = (
            dict(zip(remaining["racer_boat_number"], remaining["preview_start_timing_rank"]))
            if "preview_start_timing_rank" in remaining.columns
            else {}
        )

        if not boats:
            order[race_id] = []
            continue

        picked = [boats[0]]
        rest = boats[1:]

        c1 = courses[picked[0]]
        if use_start_rank and pattern.second_given_first_by_start_rank:
            best_start = start_rank_by_boat.get(picked[0]) == 1
            second_probs = pattern.second_given_first_by_start_rank.get(
                (c1, best_start), pattern.second_given_first.get(c1, {})
            )
        else:
            second_probs = pattern.second_given_first.get(c1, {})
        if rest and second_probs:
            second = max(
                rest,
                key=lambda b: _blend(score_by_boat, rest, b, second_probs.get(courses[b], 0.0)),
            )
            picked.append(second)
            rest = [b for b in rest if b != second]

        if len(picked) == 2:
            c2 = courses[picked[1]]
            third_probs = pattern.third_given_first_second.get((c1, c2), {})
            if rest and third_probs:
                third = max(
                    rest,
                    key=lambda b: _blend(score_by_boat, rest, b, third_probs.get(courses[b], 0.0)),
                )
                picked.append(third)
                rest = [b for b in rest if b != third]

        order[race_id] = picked + rest

    return order


def _blend(score_by_boat: dict[int, float], candidates: list[int], boat: int, pattern_prob: float) -> float:
    """モデルスコア(候補内で0〜1に正規化)と出目パターン確率を掛け合わせる。

    候補が1艇しかない場合はスコア正規化が0除算になるので素通しする。
    """
    values = [score_by_boat[b] for b in candidates]
    lo, hi = min(values), max(values)
    normalized_score = (score_by_boat[boat] - lo) / (hi - lo) if hi > lo else 1.0
    # どちらかが0だと積が0になり選ばれなくなるので、両方に小さな下駄を履かせる
    return (normalized_score + 0.01) * (pattern_prob + 0.01)
