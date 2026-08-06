"""1着になる確率をロジスティック回帰でスコアリングする最初のモデル。

艇(行)ごとに独立な二値分類(is_win)として学習する簡易的な方法。
本来は同一レース内6艇の相対関係を直接モデル化する(ランキング学習)方が
筋が良いが、まずはシンプルな基準として組む。精度が出ない場合の
次の一手として記録しておく。
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "weight",
    "flying_count",
    "late_count",
    "average_start_timing",
    "national_win_rate",
    "national_top2_rate",
    "national_top3_rate",
    "local_win_rate",
    "local_top2_rate",
    "local_top3_rate",
    "motor_top2_rate",
    "motor_top3_rate",
    "boat_top2_rate",
    "boat_top3_rate",
    "exhibition_time",
    "exhibition_time_rank",
    "preview_start_timing",
    "preview_start_timing_rank",
    "tilt_adjustment",
    "weight_adjustment",
    "wind_speed",
    "wave_height",
    "air_temperature",
    "water_temperature",
    "tide_level_cm",
    "tide_trend_cm_per_hour",
    "moon_age",
    "venue_course_win_rate",
    # racer_course_win_rate は検証の結果いったん除外(features/encodings.py 参照)。
    # 選手×コースは1人あたりのサンプル数が少なく、縮小推定を強くかけないと
    # ノイズを拾って単勝的中率が悪化した(54.1%→50.5%)。データが増えたら再検討する。
]

CATEGORICAL_FEATURES = [
    "course_number",
    "stadium_number",
    "weather_number",
    "wind_direction_number",
    "tide_name",
    "class_number",  # 選手級別(A1/A2/B1/B2)。実装当初から取得はしていたが特徴量に
    # 使うのを忘れていた(2026-08-06、ユーザー指摘で発覚)
    "water_type",  # 水質(海水/汽水/淡水)。塩分濃度で艇の浮力・スピードが変わる
]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def train(df: pd.DataFrame) -> Pipeline:
    pipeline = build_pipeline()
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_win"].astype(int)
    pipeline.fit(x, y)
    return pipeline


def predict_win_probability(pipeline: Pipeline, df: pd.DataFrame) -> pd.Series:
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return pd.Series(pipeline.predict_proba(x)[:, 1], index=df.index)
