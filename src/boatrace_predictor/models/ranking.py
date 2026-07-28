"""同一レース内の6艇を直接ランキング学習するモデル(LightGBM LambdaRank)。

models/scoring.py のロジスティック回帰は艇ごと独立な二値分類(1着か否か)
だったため、2着・3着の予測精度が保証されず、3連単/2連単のような
複数艇を当てる券種の回収率が伸び悩んだ。LambdaRankはレース内の
着順を直接の学習目標にするため、上位互換として試す。
"""

import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from boatrace_predictor.models.scoring import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
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


def _relevance(place_number: pd.Series) -> pd.Series:
    # 1着=3, 2着=2, 3着=1, 4着以下(欠場等含むNaNも)=0 の段階的な目的変数
    return (4 - place_number).clip(lower=0).fillna(0).astype(int)


def train_ranker(df: pd.DataFrame) -> tuple[ColumnTransformer, lgb.LGBMRanker]:
    """race_id ごとにグループ化したLambdaRankモデルを学習する。"""
    sorted_df = df.sort_values(["race_id", "racer_boat_number"]).reset_index(drop=True)

    preprocessor = _build_preprocessor()
    x = preprocessor.fit_transform(sorted_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    y = _relevance(sorted_df["place_number"])
    group = sorted_df.groupby("race_id", sort=False).size().tolist()

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        random_state=0,
        verbosity=-1,
    )
    ranker.fit(x, y, group=group)
    return preprocessor, ranker


def predict_scores(
    preprocessor: ColumnTransformer, ranker: lgb.LGBMRanker, df: pd.DataFrame
) -> pd.Series:
    x = preprocessor.transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    return pd.Series(ranker.predict(x), index=df.index)
