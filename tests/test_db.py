import json
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from boatrace_predictor.data.schemas import ApiV1Response
from boatrace_predictor.persistence.db import save_race
from boatrace_predictor.persistence.models import (
    Payout,
    PreviewEntry,
    PreviewWeather,
    Race,
    RacerEntry,
    RaceWeather,
    ResultEntry,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def _load_race_with_preview_and_result():
    raw = json.loads(
        (FIXTURES / "api_v1_20260720_trimmed.json").read_text(encoding="utf-8")
    )
    parsed = ApiV1Response.model_validate(raw)
    return next(
        r
        for stadium in parsed.programs.stadiums.values()
        for r in stadium.races.values()
        if r.result is not None and r.preview is not None
    )


def test_save_race_persists_racers_preview_result_and_payouts() -> None:
    engine = _engine()
    race = _load_race_with_preview_and_result()

    with Session(engine) as session:
        save_race(session, race)
        session.commit()

        races = session.exec(select(Race)).all()
        assert len(races) == 1
        db_race = races[0]

        entries = session.exec(
            select(RacerEntry).where(RacerEntry.race_id == db_race.id)
        ).all()
        assert len(entries) == 6

        preview_weather = session.exec(
            select(PreviewWeather).where(PreviewWeather.race_id == db_race.id)
        ).first()
        assert preview_weather is not None

        preview_entries = session.exec(
            select(PreviewEntry).where(PreviewEntry.race_id == db_race.id)
        ).all()
        assert len(preview_entries) == 6

        result_weather = session.exec(
            select(RaceWeather).where(RaceWeather.race_id == db_race.id)
        ).first()
        assert result_weather is not None

        result_entries = session.exec(
            select(ResultEntry).where(ResultEntry.race_id == db_race.id)
        ).all()
        assert len(result_entries) == 6

        payouts = session.exec(
            select(Payout).where(Payout.race_id == db_race.id, Payout.bet_type == "trifecta")
        ).all()
        assert len(payouts) >= 1


def test_save_race_skips_payout_entries_with_null_combination_or_amount() -> None:
    """稀に payouts の combination/amount が null になることがある(2026-08-22確認)。
    保存時にクラッシュせず、そのエントリだけスキップされることを確認する。"""
    engine = _engine()
    race = _load_race_with_preview_and_result()
    assert race.result is not None and race.result.payouts is not None

    from boatrace_predictor.data.schemas import PayoutEntry

    race.result.payouts.win = [
        PayoutEntry(combination="1", amount=150),
        PayoutEntry(combination=None, amount=200),
        PayoutEntry(combination="2", amount=None),
    ]

    with Session(engine) as session:
        save_race(session, race)
        session.commit()

        win_payouts = session.exec(
            select(Payout).where(Payout.bet_type == "win")
        ).all()
        assert len(win_payouts) == 1
        assert win_payouts[0].combination == "1"
        assert win_payouts[0].amount == 150


def test_save_race_is_idempotent_on_rerun() -> None:
    engine = _engine()
    race = _load_race_with_preview_and_result()

    with Session(engine) as session:
        save_race(session, race)
        session.commit()
        save_race(session, race)
        session.commit()

        races = session.exec(select(Race)).all()
        assert len(races) == 1

        entries = session.exec(
            select(RacerEntry).where(RacerEntry.race_id == races[0].id)
        ).all()
        assert len(entries) == 6
