"""
Reddit PA Pipeline.

Online learning cho Reddit post popularity prediction.
Dùng PA-I regression + PA classification với uncertainty weighting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..input.feature_builder import RedditFeatureBuilder
from .target_builder import RedditTargetBuilder, compute_reddit_score
from .temporal_sampler import RedditTemporalSampler
from ...pa_core import QuadRegressionHead, PAClassificationHead
from ...pa_core.regression_head import TARGET_NAMES as _DEFAULT_TARGET_NAMES


_REDDIT_TARGET_NAMES = ["score", "num_comments", "upvote_ratio"]

LABEL_NAMES = {0: "Low", 1: "Medium", 2: "Popular", 3: "Viral"}


class RedditPAPipeline:
    """
    PA online learning pipeline cho Reddit.

    Input columns:
        title       : str       – tiêu đề bài đăng
        selftext    : str       – nội dung (có thể rỗng)
        subreddit   : str       – tên subreddit
        score       : int       – số upvotes - downvotes
        num_comments: int       – số bình luận
        upvote_ratio: float     – tỉ lệ upvote [0, 1]
        created_utc : int/float – UNIX timestamp

    Usage
    -----
    pipeline = RedditPAPipeline()
    pipeline.fit(df_train)
    results = pipeline.predict(df_test)
    """

    def __init__(
        self,
        title_dim: int = 32,
        body_dim: int = 32,
        subreddit_dim: int = 16,
        regression_C: float = 1.0,
        classification_C: float = 1.0,
        epsilon: float = 0.1,
        add_engagement: bool = False,
    ):
        self.feature_builder = RedditFeatureBuilder(
            title_dim=title_dim,
            body_dim=body_dim,
            subreddit_dim=subreddit_dim,
            add_engagement=add_engagement,
        )
        self.target_builder = RedditTargetBuilder()
        self.regression_head = QuadRegressionHead(
            C=regression_C, epsilon=epsilon,
            target_names=_REDDIT_TARGET_NAMES,
        )
        self.classification_head = PAClassificationHead(
            C=classification_C, n_classes=4,
            label_names=LABEL_NAMES,
        )
        self._trained = False

    def fit(self, df: pd.DataFrame) -> "RedditPAPipeline":
        Z = self.feature_builder.fit_transform(df)
        Y_reg, y_cls, _ = self.target_builder.fit_transform(df)

        self.regression_head.run_online(Z, Y_reg)
        self.classification_head.run_online(Z, y_cls)

        self._trained = True
        return self

    def fit_temporal(self, df: pd.DataFrame) -> "RedditPAPipeline":
        sampler = RedditTemporalSampler()
        df_t, df_t1 = sampler.create_pairs(df)

        self.feature_builder.add_engagement = True
        Z = self.feature_builder.fit_transform(df_t)

        self.target_builder.fit(df_t1)
        Y_reg = self.target_builder.regression_targets(df_t1)
        y_cls = self.target_builder.classification_labels(df_t1)

        self.regression_head.run_online(Z, Y_reg)
        self.classification_head.run_online(Z, y_cls)

        self._trained = True
        self._temporal_sampler = sampler
        return self

    def partial_fit(self, df: pd.DataFrame) -> "RedditPAPipeline":
        if not self._trained:
            return self.fit(df)

        Z = self.feature_builder.transform(df)
        Y_reg = self.target_builder.regression_targets(df)
        y_cls = self.target_builder.classification_labels(df)

        for i in range(len(Z)):
            x = Z[i: i + 1]
            for j, name in enumerate(_REDDIT_TARGET_NAMES):
                self.regression_head.models[name].partial_fit(x, [float(Y_reg[i, j])])
            self.classification_head.model.partial_fit(x, [int(y_cls[i])])

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._trained:
            raise RuntimeError("Call fit() before predict().")

        Z = self.feature_builder.transform(df)
        reg_preds = self.regression_head.predict(Z)
        cls_preds = self.classification_head.predict(Z)

        result = df.copy().reset_index(drop=True)
        result["pred_score"]        = reg_preds["score"]
        result["pred_num_comments"] = reg_preds["num_comments"]
        result["pred_upvote_ratio"] = reg_preds["upvote_ratio"]
        result["pred_label"]        = cls_preds
        result["pred_label_name"]   = [
            self.classification_head.label_name(int(l)) for l in cls_preds
        ]
        result["predicted_popularity_score"] = compute_reddit_score(
            reg_preds["score"],
            reg_preds["num_comments"],
            reg_preds["upvote_ratio"],
        )
        return result

    def predict_single(self, row: dict) -> dict:
        return self.predict(pd.DataFrame([row])).iloc[0].to_dict()

    @property
    def regression_history(self):
        return self.regression_head.history

    @property
    def classification_history(self):
        return self.classification_head.history

    @property
    def label_thresholds(self):
        return self.target_builder.label_thresholds
