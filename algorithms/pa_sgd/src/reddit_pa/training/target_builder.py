"""
Reddit TargetBuilder.

Regression targets (log1p-transformed):
    score, num_comments, upvote_ratio

Popularity score:
    S = log(max(score, 0) + 1) + 3·log(num_comments + 1) + log(upvote_ratio·100 + 1)

Classification labels (4-class, percentile-based):
    Label 3 (Viral)   : Top 5%  (p95)
    Label 2 (Popular) : Top 5%–20% (p80–p95)
    Label 1 (Medium)  : Top 20%–50% (p50–p80)
    Label 0 (Low)     : Bottom 50%
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reddit_score(
    score: np.ndarray | float,
    num_comments: np.ndarray | float,
    upvote_ratio: np.ndarray | float,
) -> np.ndarray | float:
    """
    Reddit popularity score.

    Trọng số:
    - comments nhân 3: community engagement quan trọng hơn upvotes
    - upvote_ratio × 100 → scale tương đương score
    """
    return (
        np.log1p(np.maximum(score, 0))
        + 3.0 * np.log1p(np.maximum(num_comments, 0))
        + np.log1p(np.maximum(upvote_ratio, 0) * 100.0)
    )


class RedditTargetBuilder:
    """
    Tạo regression targets và classification labels từ Reddit data.

    Required columns: score, num_comments, upvote_ratio
    """

    REGRESSION_COLS = ["score", "num_comments", "upvote_ratio"]
    TARGET_NAMES    = ["score", "num_comments", "upvote_ratio"]

    COL_ALIASES: dict[str, list[str]] = {
        "score":        ["ups", "upvotes"],
        "num_comments": ["comments", "comment_count"],
        "upvote_ratio": ["ratio", "upvote_rate"],
    }

    def __init__(self) -> None:
        self._p95: float | None = None
        self._p80: float | None = None
        self._p50: float | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "RedditTargetBuilder":
        scores = self._raw_popularity_scores(df)
        self._p95 = float(np.percentile(scores, 95))
        self._p80 = float(np.percentile(scores, 80))
        self._p50 = float(np.percentile(scores, 50))
        self._fitted = True
        return self

    def regression_targets(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns log1p-transformed regression targets.
        Shape: (n, 3) — columns: [score, num_comments, upvote_ratio]
        Note: upvote_ratio is already in [0,1], so log1p(ratio) works fine.
        """
        df = self._normalize_cols(df)
        return np.column_stack([
            np.log1p(df["score"].clip(lower=0).values),
            np.log1p(df["num_comments"].clip(lower=0).values),
            np.log1p(df["upvote_ratio"].clip(0, 1).values),
        ]).astype(np.float32)

    def classification_labels(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before classification_labels().")
        scores = self._raw_popularity_scores(df)
        return self._apply_thresholds(scores)

    def popularity_scores(self, df: pd.DataFrame) -> np.ndarray:
        return self._raw_popularity_scores(df)

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.fit(df)
        y_reg   = self.regression_targets(df)
        y_score = self._raw_popularity_scores(df)
        y_cls   = self._apply_thresholds(y_score)
        return y_reg, y_cls, y_score.astype(np.float32)

    @property
    def label_thresholds(self) -> dict[str, float]:
        return {"p50": self._p50, "p80": self._p80, "p95": self._p95}

    def _normalize_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for std_name, aliases in self.COL_ALIASES.items():
            if std_name not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        df = df.rename(columns={alias: std_name})
                        break
            if std_name not in df.columns:
                df[std_name] = 0
        return df

    def _raw_popularity_scores(self, df: pd.DataFrame) -> np.ndarray:
        df = self._normalize_cols(df)
        return compute_reddit_score(
            df["score"].clip(lower=0).values,
            df["num_comments"].clip(lower=0).values,
            df["upvote_ratio"].clip(0, 1).values,
        ).astype(np.float32)

    def _apply_thresholds(self, scores: np.ndarray) -> np.ndarray:
        labels = np.zeros(len(scores), dtype=np.int64)
        labels[scores >= self._p50] = 1
        labels[scores >= self._p80] = 2
        labels[scores >= self._p95] = 3
        return labels
