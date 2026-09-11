"""当日(または指定日)の出走表・直前情報を取得し、モデルで予想してDBにログ保存する。

実際の舟券は買わない(ペーパー予想運用、CLAUDE.mdフェーズ5)。予想対象日より
前の結果データだけでモデルを学習するので、未来のデータを混ぜていない。
翌日以降に scripts/daily_report.py で結果と突き合わせて精度を確認する。

2026-08-22: 公式サイトのオッズページをPlaywrightで取得し、期待値(予想確率×オッズ)
を計算してログに残すようにした。ただし4〜6号艇の高確率帯でモデルが過大評価する
傾向が判明したため「観察のみ」モード: [参考・未検証]として表示するだけで、
行動の判断材料としては使わない(EV_THRESHOLD付近のコメント参照)。
オッズ取得は公式サイトへの直接アクセスなので、失敗しても予想自体は止めない
(--skip-oddsで無効化もできる)。

発走時刻(closed_at)を過ぎたレースはデフォルトでスキップする。直前情報は
API側で発走後にも更新されることがあり、発走済みレースに対して予想を作ると
「本当にレース前に分かっていた情報」ではなくなってしまうため
(2026-08-03、7Rの予想が再現できない問題が発生して判明)。
過去日を指定して研究目的で無理やり予想を作りたい場合は --allow-finished を使う。

同じ理由で、展示タイム・展示STがまだ公式に出ていないレースも予想を保存しない
(2026-09-03発覚: 本番ログの80.6%が展示タイム欠損のまま確定しており、これが
バックテストで確認した単勝的中率54〜58%を実運用が大きく下回っていた主因と
判明した)。データが揃うまで次回実行を待ち、それでも発走まで揃わなかった
レースは予想なしのまま残る(不完全な予想を残すより正直な状態)。

このスクリプトは1日1回ではなく、レース開催中は30分〜1時間おきに繰り返し
実行する運用を想定している(1回で全レース分の直前情報が揃うわけではないため)。
既に予想済みのレースは再度上書きされる(直前情報が更新され次第、最新の内容で
上書きする設計)。

2026-09-09: GitHub Actions移行後、外部からの起動頻度(schedule:)がGitHub側の
都合で数時間おきにしか実際には発火せず、「直前情報がまだ無い」→(次に実行される
頃には)「もう発走済み」という間を縫ってしまい、丸1日予想0件という日が発生した。
外部トリガーを増やす(cron-job.org等)以外の対処として、1回の実行内で
未確定レースが無くなるかタイムアウトするまで数分おきに内部リトライするように
した(_POLL_INTERVAL_SECONDS/_MAX_POLL_SECONDS)。

2026-09-11: 18分のポーリングでもまだカバー不足(1日通して数レースしか拾えない日が
あった)と判明。GitHub Actionsの無料枠は1ジョブ最大6時間動けるので、
_MAX_POLL_SECONDSを最大5時間台まで伸ばし、「朝に1回でも起動できれば、その1回の
実行だけでほぼ1日分をポーリングでカバーする」設計に変更した。scheduleの発火が
不安定でも、1日1〜2回捕まえられれば十分になる。GitHub Actions側のtimeout-minutes
もこれに合わせて延長する。

使い方:
    python scripts/daily_predict.py
    python scripts/daily_predict.py --date 2026-07-29 --allow-finished
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger
from playwright.sync_api import sync_playwright
from sqlmodel import Session

# Boatrace Open APIのclosed_at等はJST(タイムゾーン情報なしの文字列)で来る。
# ローカルWindows PC(JST設定)ではdatetime.now()がたまたま一致していたため
# 気づかなかったが、GitHub Actions移行(2026-09-04、Ubuntuランナー=UTC)後に
# 「発走済みでスキップ」が常に0件になる不具合として発覚(2026-09-09)。
# UTC 5:27の実行で closed_at 10:30 のレースを「まだ発走前」と誤判定していた
# (UTCの5:27をJSTのclosed_atとそのまま比較していたため、実際には9時間以上前に
# 発走済みだった)。実行環境のタイムゾーンに関係なく正しく比較できるよう、
# 「今」は常にJSTとして明示的に計算する。
_JST = ZoneInfo("Asia/Tokyo")


def _now_jst() -> datetime:
    return datetime.now(_JST).replace(tzinfo=None)


# 2026-09-11: 1回の実行が最大5時間40分粘るようになったが、GitHub Releaseへの
# DBアップロードはワークフロー側の最後のステップでしか行われないため、実行中は
# 進捗が全く外から見えず、350分のジョブタイムアウトで強制終了した場合はその日の
# 進捗が丸ごと失われるリスクがあった。ポーリングの各周回ごとにこの関数で
# 直接アップロードし、いつでも進捗を確認できる・失われないようにする。
# GitHub Actions上でのみ動く(GITHUB_ACTIONS環境変数で判定、ローカル実行では何もしない)。
def _upload_progress_in_ci(db_path: Path) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return
    try:
        subprocess.run(
            ["gh", "release", "upload", "db-latest", str(db_path), "--repo", repo, "--clobber"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as e:  # 進捗アップロードの失敗で本体の処理を止めない
        logger.warning(f"進捗アップロード失敗: {e}")


# 2026-09-01: タスクスケジューラのS4U化(ログイン無しでも起動できる)は解決したが、
# 別の問題として「実行の途中でPCがスリープして数時間止まる」ことが判明した
# (8/31、開始から4時間41分後にログが再開する形跡があった)。実行中はWindowsに
# 「システムをスリープさせないでほしい」と明示的にリクエストする(電源設定自体は
# 変更しない、このプロセスが動いている間だけの一時的な要求)。
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def _prevent_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    except (AttributeError, OSError):
        yield  # Windows以外(開発機がmac/linuxの場合など)では何もしない
        return
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


from boatrace_predictor.config import STADIUMS, settings
from boatrace_predictor.data.client import BoatraceOpenAPIClient
from boatrace_predictor.data.odds_client import OddsClient
from boatrace_predictor.data.parts_exchange_client import PartsExchangeClient
from boatrace_predictor.features.dataset import build_dataset, build_race_features
from boatrace_predictor.features.encodings import apply_course_encodings, fit_course_encodings
from boatrace_predictor.features.finish_pattern import fit_finish_pattern, rerank_with_pattern
from boatrace_predictor.models import ranking
from boatrace_predictor.models.scoring import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    predict_win_probability,
    train,
)
from boatrace_predictor.persistence.db import (
    engine,
    init_db,
    save_odds,
    save_parts_exchange,
    save_prediction,
    save_race,
)

# 2026-08-06: LightGBMランキングモデルに切り替え(単勝/複勝の回収率で
# ロジスティック回帰を上回るようになったため)。ロジスティック回帰は
# 比較用に引き続きログし続ける。PRIMARY_MODELの予想だけを標準出力に表示する。
PRIMARY_MODEL = "lightgbm_ranking"

# 期待値(予想確率×オッズ)がこの値以上なら「買いシグナル」として表示する。
# 1.0が理論上の損益分岐点。ただし4〜6号艇の高確率帯でモデルが過大評価する
# 傾向が確認されており(2026-08-22)、この値を超えても実際にプラスとは限らない。
# 「観察のみ」モード: 参考表示するだけで、行動の判断材料には使わないこと。
EV_THRESHOLD = 1.2

# 2026-09-09/11: 直前情報がまだ揃っていないレースを、揃うまでこのスクリプト内で
# 待ってリトライする際の間隔と上限時間。GitHub Actionsの無料枠は1ジョブ最大6時間
# 動けるため、5時間40分(セットアップ・DBアップロード分の余裕を見て)まで粘る。
# これにより、schedule:の発火が1日1回でも成功すれば、その1回でほぼ1日分の
# レースをカバーできる設計にした(2026-09-11、18分だと拾えるレースが少なすぎた)。
_POLL_INTERVAL_SECONDS = 180
_MAX_POLL_SECONDS = 340 * 60


def _fit_models(history):
    """(model_name, predict_fn) のリストを返す。predict_fn(df) -> pd.Series[win_prob的なスコア]"""
    logistic_model = train(history)
    preprocessor, ranker = ranking.train_ranker(history)
    return [
        (PRIMARY_MODEL, lambda df: ranking.predict_scores(preprocessor, ranker, df)),
        ("logistic_regression", lambda df: predict_win_probability(logistic_model, df)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD(未指定なら本日)")
    parser.add_argument("--stadiums", type=int, nargs="*", default=settings.target_stadiums)
    parser.add_argument(
        "--allow-finished",
        action="store_true",
        help="発走済みレースも予想対象にする(研究目的の過去日再現用。通常は使わない)",
    )
    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="公式サイトのオッズ取得(Playwright)を無効化する",
    )
    parser.add_argument(
        "--skip-parts",
        action="store_true",
        help="公式サイトの部品交換情報取得(Playwright)を無効化する",
    )
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else _now_jst().date()
    stadiums = set(args.stadiums)

    init_db()

    with Session(engine) as session:
        # 1. target_dateより前の結果データだけで学習する(未来データを混ぜない)
        history = build_dataset(
            session,
            stadium_numbers=args.stadiums,
            date_to=(target_date - timedelta(days=1)).isoformat(),
        )
        history = history[history["is_normal_finish"]].reset_index(drop=True)
        if history.empty or history["is_win"].sum() == 0:
            logger.error(f"{target_date}より前の学習データが無いため予想できません")
            return

        encodings = fit_course_encodings(history)
        history = apply_course_encodings(history, encodings)
        # 2着・3着はモデルのスコア順に並べるだけでは艇同士の関係(出目の偏り)を
        # 無視することになるため、1着コースから見た2着・3着コースの経験分布で
        # 組み直す(features/finish_pattern.py)。2026-08-23、YouTube調査をきっかけに
        # 検証: 2連単の的中率が19.8%→21.6%、2連複が27.7%→29.9%に改善(train_frac
        # 0.5〜0.8の複数分割で再現確認済み)。1着の予想自体は変えない。
        finish_pattern = fit_finish_pattern(history)
        models = _fit_models(history)
        logger.info(
            f"学習データ: {history['race_id'].nunique()}レース"
            f"({history['date'].min()}〜{history['date'].max()})"
        )

        # OddsClientとPartsExchangeClientはどちらも公式サイトをPlaywrightで開くが、
        # sync_playwright()を同一プロセスで2回起動すると衝突する(2026-08-24判明)ため、
        # ブラウザを1つだけ起動して両方に共有させる。ポーリングループ全体で使い回す。
        need_browser = not (args.skip_odds and args.skip_parts)
        playwright_ctx = sync_playwright().start() if need_browser else None
        browser = playwright_ctx.chromium.launch(headless=True) if playwright_ctx else None
        odds_client = None if args.skip_odds else OddsClient(browser=browser).__enter__()
        parts_client = None if args.skip_parts else PartsExchangeClient(browser=browser).__enter__()

        predicted_race_ids: set[int] = set()
        n_skipped_finished = 0
        n_skipped_incomplete = 0
        n_ev_observed = 0
        poll_start = time.monotonic()

        try:
            while True:
                # 2. 当日データ取得・保存(出走表+直前情報。結果はまだ無い想定)
                with BoatraceOpenAPIClient() as client:
                    races = client.fetch_day(target_date, stadiums=stadiums)
                if not races:
                    logger.warning(f"{target_date}: 対象会場のレースが見つかりませんでした")
                    return

                race_ids = {}
                for race in races:
                    race_id = save_race(session, race)
                    race_ids[(race.stadium_number, race.race_number)] = race_id
                session.commit()

                # 3. レースごとに予想してログ保存(発走済みは原則スキップ)
                now = _now_jst()
                predicted_at = now.isoformat(timespec="seconds")
                n_skipped_finished = 0
                n_skipped_incomplete = 0

                for race in races:
                    if race.closed_at and not args.allow_finished:
                        closed_at = datetime.strptime(race.closed_at, "%Y-%m-%d %H:%M:%S")
                        if closed_at <= now:
                            n_skipped_finished += 1
                            continue

                    race_id = race_ids[(race.stadium_number, race.race_number)]
                    df = build_race_features(session, race_id)
                    if df["national_win_rate"].isna().all():
                        continue  # 出走表自体が無い(会場コードの取り違え等)場合はスキップ

                    # 2026-09-03: 展示タイム・展示STは発走が近づくまで公式に出ないことがあり、
                    # 未確定のまま予想を確定させると欠損値が中央値補完されて精度が落ちる
                    # (ログ調査の結果、本番予想の80.6%が展示タイム欠損のまま保存されており、
                    # 実際に単勝的中率が過去のバックテスト水準54〜58%を大きく下回る一因と
                    # 判明した)。欠けている間は予想を保存せず、次回実行(揃った時点)を待つ。
                    # 発走直前まで一度も揃わなかったレースは結果として予想なしのままになるが、
                    # 不完全なデータでの予想を「最終版」として残すよりは正直な状態。
                    if not args.allow_finished and (
                        df["exhibition_time"].isna().any()
                        or df["preview_start_timing"].isna().any()
                    ):
                        n_skipped_incomplete += 1
                        continue

                    df = apply_course_encodings(df, encodings)
                    features_snapshot = df[
                        ["racer_boat_number"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
                    ].to_dict("records")

                    primary_ranked = None
                    primary_scores_series = None
                    calibrated_win_prob = None  # ロジスティック回帰の確率(EV計算用、検証済み)
                    for model_name, predict_fn in models:
                        scores_series = predict_fn(df)
                        ranked = rerank_with_pattern(df, scores_series, finish_pattern)[race_id]
                        scores = dict(zip(df["racer_boat_number"], scores_series))
                        save_prediction(
                            session,
                            race_id,
                            predicted_at,
                            model_name,
                            ranked,
                            scores,
                            features_snapshot,
                        )
                        if model_name == PRIMARY_MODEL:
                            primary_ranked = ranked
                            primary_scores_series = scores_series
                        if model_name == "logistic_regression":
                            calibrated_win_prob = scores
                    predicted_race_ids.add(race_id)

                    name = STADIUMS[race.stadium_number]
                    primary_scores = dict(zip(df["racer_boat_number"], primary_scores_series))
                    # LightGBMランキングモデルのスコアは確率ではない(0〜1に収まらない生スコア)ので
                    # %表示はしない。ロジスティック回帰のような確率が欲しい場合は
                    # daily_report.py --model logistic_regression 側のログを見る
                    top1_score, top2_score = sorted(primary_scores.values(), reverse=True)[:2]
                    margin = top1_score - top2_score
                    # 2026-08-23: 1位・2位のスコア差(margin)でレースを4分位に分けてバックテスト
                    # したところ、marginが大きいほど単勝的中率が上がる(36.7%→72.6%)ことを確認。
                    # ただし単発の単勝回収率はほぼ変わらない(市場のオッズが自信度をある程度
                    # 織り込んでいるため、favorite-longshot bias)。一方、転がし(korogashi)は
                    # 連続的中が必要なので、的中率が上がること自体に価値がある
                    # (的中率0.726のレースだけを5連続なら生存率20.4%、0.54なら4.6%)。
                    # そのためEVシグナルとは別に「参考: 高確信」という表示だけ追加する
                    # (行動を強制するものではなく、転がし対象レースを絞る判断材料)。
                    confidence_tag = " [参考:高確信]" if margin >= 1.73 else ""
                    print(
                        f"{race.date} {name} {race.race_number}R: "
                        f"予想 {primary_ranked[0]}-{primary_ranked[1]}-{primary_ranked[2]} "
                        f"(1着スコア: {top1_score:.2f}, margin: {margin:.2f}){confidence_tag}"
                    )

                    if odds_client is not None:
                        try:
                            win_odds, place_odds = odds_client.fetch_odds(
                                race.stadium_number, target_date, race.race_number
                            )
                        except Exception as e:  # スクレイピング失敗は予想自体を止めない
                            logger.warning(f"  オッズ取得失敗: {e}")
                            win_odds, place_odds = {}, {}

                        if win_odds:
                            save_odds(session, race_id, predicted_at, win_odds, place_odds)
                            # 期待値計算にはロジスティック回帰の確率を使う(キャリブレーション
                            # 検証済み: 予測58.0% vs 実際56.3%と実績に近い)。LightGBMの生スコアは
                            # ランキング最適化のためのもので、softmaxしても実際の勝率とは
                            # 一致しない可能性が高く、期待値計算には不適切と判断した(2026-08-22)。
                            #
                            # 2026-08-22: 4〜6号艇の高確率帯(予測30〜50%)で実際の勝率が
                            # 大幅に下回るキャリブレーションの歪みが判明(例: 予測34.5%→実際13.6%)。
                            # 「買いシグナル」として自信満々に提示するのは無責任と判断し、
                            # 「観察のみ」モードに変更。ラベルを外し参考値として記録するだけにする。
                            # もっとデータが溜まってキャリブレーション補正ができるまでは、
                            # ここに出る数字を実際の判断材料にしないこと。
                            win_prob = calibrated_win_prob or {}
                            best_boat, best_ev = None, 0.0
                            for boat, odds in win_odds.items():
                                ev = win_prob.get(boat, 0.0) * odds
                                if ev > best_ev:
                                    best_boat, best_ev = boat, ev
                            if best_boat is not None and best_ev >= EV_THRESHOLD:
                                n_ev_observed += 1
                                print(
                                    f"  [参考・未検証] {best_boat}号艇 "
                                    f"(予想確率{win_prob[best_boat]:.1%} x オッズ{win_odds[best_boat]:.1f}倍 "
                                    f"= 期待値{best_ev:.2f}) ※行動材料にはまだ使えません"
                                )

                    if parts_client is not None:
                        try:
                            parts = parts_client.fetch_parts_exchange(
                                race.stadium_number, target_date, race.race_number
                            )
                        except Exception as e:  # スクレイピング失敗は予想自体を止めない
                            logger.warning(f"  部品交換情報取得失敗: {e}")
                            parts = {}
                        if parts:
                            save_parts_exchange(session, race_id, predicted_at, parts)
                            # 5点改善計画の項目3(2026-08-23)。まだ予測モデルの特徴量には
                            # 組み込んでいない(過去データが無く検証できないため、保存だけ
                            # 先に始めてデータが溜まってから効果を検証する)。参考表示のみ。
                            parts_desc = ", ".join(
                                f"{b}号艇:{'/'.join(p)}" for b, p in parts.items()
                            )
                            print(f"  [参考] 部品交換あり: {parts_desc}")

                session.commit()
                _upload_progress_in_ci(settings.db_path)

                if n_skipped_incomplete == 0:
                    break  # 全レース処理済み(予想済みか発走済み)、ポーリング終了
                elapsed = time.monotonic() - poll_start
                if elapsed + _POLL_INTERVAL_SECONDS > _MAX_POLL_SECONDS:
                    logger.info(
                        f"直前情報未確定のレースが{n_skipped_incomplete}件残っていますが"
                        "、時間切れのためポーリングを終了します(次回実行を待ちます)"
                    )
                    break
                logger.info(
                    f"直前情報未確定のレースが{n_skipped_incomplete}件残っています。"
                    f"{_POLL_INTERVAL_SECONDS}秒後に再チェックします"
                )
                time.sleep(_POLL_INTERVAL_SECONDS)
        finally:
            if odds_client is not None:
                odds_client.__exit__(None, None, None)
            if parts_client is not None:
                parts_client.__exit__(None, None, None)
            if browser is not None:
                browser.close()
            if playwright_ctx is not None:
                playwright_ctx.stop()

        session.commit()

        logger.info(
            f"完了: {len(predicted_race_ids)}レース分の予想を保存"
            f"(発走済みでスキップ: {n_skipped_finished}件、"
            f"直前情報未確定でスキップ: {n_skipped_incomplete}件、"
            f"参考EV表示: {n_ev_observed}件(未検証))"
        )


if __name__ == "__main__":
    with _prevent_sleep():
        main()
