"""今日のこれから発走するレースに、検証中のフロー(モデル20%+市場80%の混合確率 x 現在の2連単オッズ)を
1回適用して、期待値の高い並びを一覧表示する(お金は賭けないペーパー予想)。

daily_predict.pyと同じ学習・コース別補正でlogistic回帰の勝率を出し、Plackett-Luceで2連単化、
公式の2連単オッズ(全30通り)と対数で混ぜる。DBは BOATRACE_DB_PATH で指定する(本番DBのコピーを推奨。
出走表などを書き込むため)。

使い方:
    BOATRACE_DB_PATH=/path/copy.db python scripts/live_picks.py [--min-ev 1.1] [--date YYYY-MM-DD]
"""

import argparse
import itertools
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sqlmodel import Session

from boatrace_predictor.config import STADIUMS, settings
from boatrace_predictor.data.client import BoatraceOpenAPIClient
from boatrace_predictor.data.exacta_odds_client import fetch_exacta_odds
from boatrace_predictor.features.calibration import apply_course_calibration, fit_course_calibration
from boatrace_predictor.features.dataset import build_dataset, build_race_features
from boatrace_predictor.features.encodings import apply_course_encodings, fit_course_encodings
from boatrace_predictor.models.scoring import predict_win_probability, train
from boatrace_predictor.persistence.db import engine, init_db, save_race

A = 0.2
PAIRS = list(itertools.permutations(range(1, 7), 2))
NAMES = [f"{a}-{b}" for a, b in PAIRS]
JST = timezone(timedelta(hours=9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--min-ev", type=float, default=1.1)
    ap.add_argument("--loop", action="store_true", help="終日回し、各レースの締切の約1分前(90秒前〜20秒前の窓)に1回だけ判定して--logに追記する")
    ap.add_argument("--log", default="live_picks.jsonl")
    args = ap.parse_args()
    now = datetime.now(JST).replace(tzinfo=None)
    target = date.fromisoformat(args.date) if args.date else now.date()
    init_db()
    with Session(engine) as session:
        raw = build_dataset(session, stadium_numbers=list(STADIUMS), date_to=(target - timedelta(days=1)).isoformat())
        raw = raw[raw["is_normal_finish"]].reset_index(drop=True)
        dates = sorted(raw["date"].unique())
        cut = dates[int(len(dates) * 0.8)]
        train_raw, calib_raw = raw[raw["date"] < cut].reset_index(drop=True), raw[raw["date"] >= cut].reset_index(drop=True)
        enc = fit_course_encodings(train_raw)
        hist, calib_df = apply_course_encodings(train_raw, enc), apply_course_encodings(calib_raw, enc)
        model = train(hist)
        cal = fit_course_calibration(calib_df, predict_win_probability(model, calib_df))
        print(f"学習: {hist['race_id'].nunique()}レース({hist['date'].min()}〜{hist['date'].max()})", flush=True)

        decided: set[int] = set()
        while True:
            try:
                with BoatraceOpenAPIClient() as client:
                    races = client.fetch_day(target)
            except Exception as e:  # noqa: BLE001  一時的なネット断で落ちないようにする
                print(f"出走表の取得に失敗(30秒後に再試行): {e!r}"[:150], flush=True)
                time.sleep(15)
                continue
            now = datetime.now(JST).replace(tzinfo=None)
            rows = []
            n_wait = n_done = n_later = 0
            for race in races:
                if not race.closed_at:
                    continue
                closed = datetime.strptime(race.closed_at, "%Y-%m-%d %H:%M:%S")
                rid = save_race(session, race)
                session.commit()
                if closed <= now:
                    n_done += 1
                    continue
                if args.loop and (rid in decided or not (20 <= (closed - now).total_seconds() <= 90)):
                    n_later += 1
                    continue
                df = build_race_features(session, rid)
                if df["national_win_rate"].isna().all():
                    continue
                if df["exhibition_time"].isna().any() or df["preview_start_timing"].isna().any():
                    n_wait += 1  # 展示情報がまだ揃っていない
                    continue
                df = apply_course_encodings(df, enc)
                sc = apply_course_calibration(df, predict_win_probability(model, df), cal)
                p = dict(zip(df["racer_boat_number"], sc))
                if len(p) != 6:
                    continue
                pv = np.array([max(p[i], 1e-6) for i in range(1, 7)])
                pv /= pv.sum()
                try:
                    o = fetch_exacta_odds(race.stadium_number, str(target), race.race_number)
                except Exception as e:  # noqa: BLE001
                    print(f"オッズ取得失敗 {STADIUMS[race.stadium_number]}{race.race_number}R: {e}", flush=True)
                    continue
                if not o or len(o) < 30 or any(v <= 1.0 for v in o.values()):
                    continue
                decided.add(rid)
                od = np.array([o[n] for n in NAMES])
                pm = np.array([pv[a - 1] * pv[b - 1] / (1 - pv[a - 1]) for a, b in PAIRS])
                pm /= pm.sum()
                q = 1 / od
                q /= q.sum()
                pb = np.exp(A * np.log(pm + 1e-12) + (1 - A) * np.log(q))
                pb /= pb.sum()
                for j, nm in enumerate(NAMES):
                    ev = pb[j] * od[j]
                    if ev >= args.min_ev:
                        rows.append((closed, STADIUMS[race.stadium_number], race.race_number, nm, od[j], pb[j], ev))
                        if args.loop:
                            with open(args.log, "a") as f:
                                f.write(json.dumps({"race_id": rid, "date": str(target), "stadium": race.stadium_number,
                                                    "race": race.race_number, "comb": nm, "odds": float(od[j]),
                                                    "prob": float(pb[j]), "ev": float(ev), "decided_at": now.isoformat()}) + "\n")
            print(f"{now:%H:%M} 発走済み{n_done}R / 展示情報待ち{n_wait}R / 先のレース{n_later}R / 買い候補{len(rows)}点", flush=True)
            for closed, st, rn, nm, od, pb, ev in sorted(rows):
                print(f"  {closed:%H:%M} {st} {rn}R  {nm}  オッズ{od:.1f}倍  混合確率{pb:.1%}  期待値{ev:.2f}", flush=True)
            if not args.loop or n_done == len([r for r in races if r.closed_at]):
                break
            time.sleep(15)


if __name__ == "__main__":
    main()
