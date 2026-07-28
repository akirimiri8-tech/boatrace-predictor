import pandas as pd
from sqlmodel import Session, SQLModel, create_engine

from boatrace_predictor.backtest.bet_types import evaluate_bet_types
from boatrace_predictor.persistence.models import Payout


def _engine_with_one_race_payouts() -> tuple:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        # race_id=1: 着順 1-4-5(3連単), 単勝は1
        session.add(Payout(race_id=1, bet_type="win", combination="1", amount=150))
        session.add(Payout(race_id=1, bet_type="place", combination="1", amount=110))
        session.add(Payout(race_id=1, bet_type="place", combination="4", amount=180))
        session.add(Payout(race_id=1, bet_type="exacta", combination="1-4", amount=500))
        session.add(Payout(race_id=1, bet_type="quinella", combination="1=4", amount=400))
        session.add(
            Payout(race_id=1, bet_type="quinella_place", combination="1=4", amount=200)
        )
        session.add(
            Payout(race_id=1, bet_type="trifecta", combination="1-4-5", amount=3000)
        )
        session.add(Payout(race_id=1, bet_type="trio", combination="1=4=5", amount=800))
        session.commit()
    return engine


def _df_with_predicted_order_1_4_5() -> tuple:
    # racer_boat_number=1が最高スコア、4が次、5が3番目になるように予想確率を作る
    df = pd.DataFrame(
        {
            "race_id": [1] * 6,
            "racer_boat_number": [1, 2, 3, 4, 5, 6],
        }
    )
    predicted_prob = pd.Series([0.9, 0.1, 0.2, 0.6, 0.5, 0.05], index=df.index)
    return df, predicted_prob


def test_evaluate_bet_types_all_hit_when_prediction_matches_result() -> None:
    engine = _engine_with_one_race_payouts()
    df, predicted_prob = _df_with_predicted_order_1_4_5()

    with Session(engine) as session:
        results = evaluate_bet_types(session, df, predicted_prob)

    by_type = {r.bet_type: r for r in results}
    assert by_type["win"].hit_rate == 1.0
    assert by_type["win"].roi == 150 / 100
    assert by_type["trifecta"].hit_rate == 1.0
    assert by_type["trifecta"].roi == 3000 / 100
    assert by_type["trio"].hit_rate == 1.0
    # quinella_place は combination "1=4" が1つだけでも的中扱いになる
    assert by_type["quinella_place"].hit_rate == 1.0


def test_evaluate_bet_types_miss_when_prediction_wrong() -> None:
    engine = _engine_with_one_race_payouts()
    df = pd.DataFrame(
        {
            "race_id": [1] * 6,
            "racer_boat_number": [1, 2, 3, 4, 5, 6],
        }
    )
    # 6号艇を1位予想にする(実際の1着は1号艇なので単勝は外れ)
    predicted_prob = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1, 0.9], index=df.index)

    with Session(engine) as session:
        results = evaluate_bet_types(session, df, predicted_prob)

    by_type = {r.bet_type: r for r in results}
    assert by_type["win"].hit_rate == 0.0
    assert by_type["win"].roi == 0.0
