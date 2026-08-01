from sqlmodel import Session, SQLModel, create_engine, select

from boatrace_predictor.backtest.bet_types import evaluate_bet_types_from_order
from boatrace_predictor.persistence.db import save_prediction
from boatrace_predictor.persistence.models import Payout, Prediction, PredictionScore, Race


def _engine_with_race_and_payout():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        race = Race(date="2026-07-29", stadium_number=24, number=1)
        session.add(race)
        session.flush()
        session.add(Payout(race_id=race.id, bet_type="win", combination="1", amount=150))
        session.commit()
        return engine, race.id


def test_save_prediction_stores_ranked_boats_and_scores() -> None:
    engine, race_id = _engine_with_race_and_payout()
    with Session(engine) as session:
        save_prediction(
            session,
            race_id=race_id,
            predicted_at="2026-07-29T09:00:00",
            model_name="logistic_regression",
            ranked_boats=[1, 4, 5, 2, 3, 6],
            scores={1: 0.4, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.15, 6: 0.05},
        )
        session.commit()

        prediction = session.exec(select(Prediction)).one()
        assert prediction.predicted_1st == 1
        assert prediction.predicted_2nd == 4
        assert prediction.predicted_3rd == 5

        scores = session.exec(
            select(PredictionScore).where(PredictionScore.prediction_id == prediction.id)
        ).all()
        assert len(scores) == 6


def test_save_prediction_replaces_previous_prediction_for_same_race_and_model() -> None:
    engine, race_id = _engine_with_race_and_payout()
    with Session(engine) as session:
        save_prediction(
            session, race_id, "2026-07-29T08:00:00", "logistic_regression", [1, 2, 3], {1: 0.5}
        )
        session.commit()
        save_prediction(
            session, race_id, "2026-07-29T09:00:00", "logistic_regression", [2, 1, 3], {2: 0.6}
        )
        session.commit()

        predictions = session.exec(select(Prediction)).all()
        assert len(predictions) == 1
        assert predictions[0].predicted_1st == 2


def test_evaluate_bet_types_from_order_matches_win_payout() -> None:
    engine, race_id = _engine_with_race_and_payout()
    with Session(engine) as session:
        results = evaluate_bet_types_from_order(session, {race_id: [1, 4, 5]})

    by_type = {r.bet_type: r for r in results}
    assert by_type["win"].hit_rate == 1.0
    assert by_type["win"].roi == 1.5
