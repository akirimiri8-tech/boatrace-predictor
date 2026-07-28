import json
from pathlib import Path

from boatrace_predictor.data.schemas import ApiV1Response

FIXTURES = Path(__file__).parent / "fixtures"


def _load() -> ApiV1Response:
    raw = json.loads(
        (FIXTURES / "api_v1_20260720_trimmed.json").read_text(encoding="utf-8")
    )
    return ApiV1Response.model_validate(raw)


def test_parse_stadiums_and_races() -> None:
    parsed = _load()
    assert len(parsed.programs.stadiums) > 0
    first_stadium = next(iter(parsed.programs.stadiums.values()))
    assert len(first_stadium.races) > 0


def test_race_has_racers_preview_and_result() -> None:
    parsed = _load()
    race = next(
        r
        for stadium in parsed.programs.stadiums.values()
        for r in stadium.races.values()
        if r.result is not None and r.preview is not None
    )
    assert len(race.racers) == 6
    assert race.preview is not None
    assert len(race.preview.racers) == 6
    assert race.result is not None
    assert race.result.payouts is not None
    assert race.result.payouts.trifecta[0].amount > 0
