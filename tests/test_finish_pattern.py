import pandas as pd

from boatrace_predictor.features.finish_pattern import fit_finish_pattern, rerank_with_pattern


def _train_df() -> pd.DataFrame:
    # コース1が1着のとき、2着は常にコース4(コース2は一度も来ない)という
    # 偏りをわざと強く作る。件数も多めにして平滑化の影響を小さくする。
    rows = []
    for race_id in range(1, 21):
        rows += [
            {"race_id": race_id, "racer_boat_number": 1, "course_number": 1, "place_number": 1},
            {"race_id": race_id, "racer_boat_number": 4, "course_number": 4, "place_number": 2},
            {"race_id": race_id, "racer_boat_number": 5, "course_number": 5, "place_number": 3},
        ]
    return pd.DataFrame(rows)


def test_second_given_first_reflects_empirical_frequency() -> None:
    pattern = fit_finish_pattern(_train_df())
    # コース1が1着の20レースで2着は常にコース4、コース2は一度も来ない
    # -> コース4の方が確率が高くなるはず
    probs = pattern.second_given_first[1]
    assert probs[4] > probs[2]


def test_rerank_keeps_first_place_but_changes_second_third() -> None:
    pattern = fit_finish_pattern(_train_df())

    # 検証用の1レース: モデルのスコアは艇2(コース2)がわずかに艇4(コース4)を
    # 上回るが、出目パターン的にはコース1が1着ならコース4が圧倒的に2着に来やすい
    # (学習データでは常にそうだった)ので、僅差のスコア差ならパターン側が勝つはず
    df = pd.DataFrame(
        [
            {"race_id": 100, "racer_boat_number": 1, "course_number": 1},
            {"race_id": 100, "racer_boat_number": 2, "course_number": 2},
            {"race_id": 100, "racer_boat_number": 3, "course_number": 3},
            {"race_id": 100, "racer_boat_number": 4, "course_number": 4},
            {"race_id": 100, "racer_boat_number": 5, "course_number": 5},
            {"race_id": 100, "racer_boat_number": 6, "course_number": 6},
        ]
    )
    scores = pd.Series([10.0, 2.0, 1.0, 1.9, 0.9, 0.5], index=df.index)

    order = rerank_with_pattern(df, scores, pattern)[100]

    assert order[0] == 1  # 1着はスコア最大のまま
    assert order[1] == 4  # 2着はスコアわずかに劣る艇4(パターンで圧倒的優位)になる
    assert set(order) == {1, 2, 3, 4, 5, 6}


def _train_df_with_start_rank() -> pd.DataFrame:
    # コース1が1着のとき: 1着艇のスタート順位が1位(best_start)なら2着はコース2、
    # そうでなければ2着はコース4、という偏りをわざと作る
    rows = []
    for race_id in range(1, 21):
        rows += [
            {
                "race_id": race_id,
                "racer_boat_number": 1,
                "course_number": 1,
                "place_number": 1,
                "preview_start_timing_rank": 1,
            },
            {
                "race_id": race_id,
                "racer_boat_number": 2,
                "course_number": 2,
                "place_number": 2,
                "preview_start_timing_rank": 2,
            },
            {
                "race_id": race_id,
                "racer_boat_number": 5,
                "course_number": 5,
                "place_number": 3,
                "preview_start_timing_rank": 3,
            },
        ]
    for race_id in range(21, 41):
        rows += [
            {
                "race_id": race_id,
                "racer_boat_number": 1,
                "course_number": 1,
                "place_number": 1,
                "preview_start_timing_rank": 2,
            },
            {
                "race_id": race_id,
                "racer_boat_number": 4,
                "course_number": 4,
                "place_number": 2,
                "preview_start_timing_rank": 1,
            },
            {
                "race_id": race_id,
                "racer_boat_number": 5,
                "course_number": 5,
                "place_number": 3,
                "preview_start_timing_rank": 3,
            },
        ]
    return pd.DataFrame(rows)


def test_second_given_first_by_start_rank_differs_by_start_rank() -> None:
    pattern = fit_finish_pattern(_train_df_with_start_rank())
    best_start_probs = pattern.second_given_first_by_start_rank[(1, True)]
    slow_start_probs = pattern.second_given_first_by_start_rank[(1, False)]
    assert best_start_probs[2] > best_start_probs[4]
    assert slow_start_probs[4] > slow_start_probs[2]


def test_rerank_use_start_rank_changes_pick_based_on_start_timing() -> None:
    pattern = fit_finish_pattern(_train_df_with_start_rank())
    df = pd.DataFrame(
        [
            {
                "race_id": 300,
                "racer_boat_number": 1,
                "course_number": 1,
                "preview_start_timing_rank": 1,
            },
            {
                "race_id": 300,
                "racer_boat_number": 2,
                "course_number": 2,
                "preview_start_timing_rank": 2,
            },
            {
                "race_id": 300,
                "racer_boat_number": 4,
                "course_number": 4,
                "preview_start_timing_rank": 3,
            },
        ]
    )
    scores = pd.Series([10.0, 1.0, 1.0], index=df.index)  # 艇2・艇4は僅差(実質同点)

    order = rerank_with_pattern(df, scores, pattern, use_start_rank=True)[300]
    # 1号艇がスタート順位1位(best_start)で勝ったケースなので、2着はコース2が有利なはず
    assert order[1] == 2


def test_rerank_falls_back_to_score_order_for_unseen_first_course() -> None:
    pattern = fit_finish_pattern(_train_df())
    df = pd.DataFrame(
        [
            {"race_id": 200, "racer_boat_number": 1, "course_number": 6},
            {"race_id": 200, "racer_boat_number": 2, "course_number": 5},
        ]
    )
    scores = pd.Series([9.0, 1.0], index=df.index)

    order = rerank_with_pattern(df, scores, pattern)[200]

    # 学習データにコース6が1着の例が無くても、ラプラス平滑化のおかげで
    # 何らかの確率が入っており、単純にスコア順に並ぶ
    assert order == [1, 2]
