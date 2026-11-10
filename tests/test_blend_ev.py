from boatrace_predictor.features.blend_ev import NAMES, blended_exacta_picks


def _uniform_odds(v=20.0):
    return {n: v for n in NAMES}


def test_incomplete_inputs_return_empty():
    assert blended_exacta_picks({1: 0.5}, _uniform_odds()) == []
    odds = _uniform_odds()
    odds["1-2"] = 1.0
    assert blended_exacta_picks({i: 1 / 6 for i in range(1, 7)}, odds) == []


def test_market_equal_to_model_gives_no_positive_ev_picks():
    probs = {i: 1 / 6 for i in range(1, 7)}
    # 全並びが同オッズで、市場=モデルなら期待値は全て同じ(30通りの合計1.0 / 30 x 20倍 = 0.67)
    assert blended_exacta_picks(probs, _uniform_odds(), min_ev=1.0) == []


def test_underpriced_combo_is_picked():
    probs = {1: 0.6, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.05, 6: 0.05}
    odds = _uniform_odds(40.0)
    odds["1-2"] = 200.0  # 本命絡みなのに高すぎるオッズ
    picks = blended_exacta_picks(probs, odds, min_ev=1.1)
    assert any(name == "1-2" for name, *_ in picks)
