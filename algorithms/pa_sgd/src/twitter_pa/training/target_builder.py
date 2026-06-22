"""
TargetBuilder: tạo regression targets và classification labels từ dữ liệu Twitter/X.

Regression targets (log1p-transformed):
    y_likes, y_comments, y_reposts, y_views

Classification label (popularity tier 0-3):
    Score = log(0.01*V + L + 5*C + 10*R + 1)
    Label 3 (Viral)   : Top 5%
    Label 2 (Popular) : Top 5% – 20%
    Label 1 (Medium)  : Top 20% – 50%
    Label 0 (Low)     : Bottom 50%
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Weights theo spec
_VIEW_WEIGHT = 0.01
_COMMENT_WEIGHT = 5.0
_REPOST_WEIGHT = 10.0

# Percentile thresholds
_THRESHOLDS = {
    3: 95,  # top 5%
    2: 80,  # top 5%-20%
    1: 50,  # top 20%-50%
    0: 0,   # bottom 50%
}


def compute_popularity_score(
    likes: np.ndarray | float,
    comments: np.ndarray | float,
    reposts: np.ndarray | float,
    views: np.ndarray | float,
) -> np.ndarray | float:
    return np.log(
        _VIEW_WEIGHT * np.maximum(views, 0)
        + np.maximum(likes, 0)
        + _COMMENT_WEIGHT * np.maximum(comments, 0)
        + _REPOST_WEIGHT * np.maximum(reposts, 0)
        + 1.0
    )


def assign_labels(scores: np.ndarray) -> np.ndarray:
    p95 = np.percentile(scores, 95)
    p80 = np.percentile(scores, 80)
    p50 = np.percentile(scores, 50)

    labels = np.zeros(len(scores), dtype=np.int64)
    labels[scores >= p80] = 2
    labels[scores >= p95] = 3
    labels[(scores >= p50) & (scores < p80)] = 1
    return labels


class TargetBuilder:

    REGRESSION_COLS = ["likes", "comments", "reposts", "views"]
    REGRESSION_TARGET_NAMES = ["y_likes", "y_comments", "y_reposts", "y_views"]

    # Tên cột thay thế khi tên cột gốc khác
    COL_ALIASES: dict[str, list[str]] = {
        "comments": ["replies", "comments_count"],
        "reposts": ["retweets", "shares", "repost_count"],
        "views": ["impressions", "view_count"],
        "likes": ["favorites", "like_count"],
    }

    def __init__(self) -> None:
        self._p95: float | None = None
        self._p80: float | None = None
        self._p50: float | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TargetBuilder":
        """
        Tính và lưu percentile thresholds từ training data.
        Phải gọi trước khi gọi classification_labels().
        """
        scores = self._raw_popularity_scores(df)
        self._p95 = float(np.percentile(scores, 95))
        self._p80 = float(np.percentile(scores, 80))
        self._p50 = float(np.percentile(scores, 50))
        self._fitted = True
        return self

    def regression_targets(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns log1p-transformed regression targets.
        Shape: (n, 4) — columns: [y_likes, y_comments, y_reposts, y_views]
        """
        df = self._normalize_cols(df)
        result = np.column_stack([
            np.log1p(df["likes"].clip(lower=0).values),
            np.log1p(df["comments"].clip(lower=0).values),
            np.log1p(df["reposts"].clip(lower=0).values),
            np.log1p(df["views"].clip(lower=0).values),
        ]).astype(np.float32)
        return result

    def classification_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns popularity labels {0, 1, 2, 3}.
        Gọi fit() trước nếu không dùng label có sẵn.
        """
        if "label" in df.columns:
            return df["label"].astype(np.int64).values
        if not self._fitted:
            raise RuntimeError("Call fit() before classification_labels().")
        scores = self._raw_popularity_scores(df)
        return self._apply_thresholds(scores)

    def popularity_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Returns raw popularity scores (không gán nhãn)."""
        return self._raw_popularity_scores(df)

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience: fit + return (y_reg, y_cls, y_score) cùng lúc.

        Returns
        -------
        y_reg   : (n, 4) float32
        y_cls   : (n,)   int64
        y_score : (n,)   float32
        """
        self.fit(df)
        y_reg = self.regression_targets(df)
        y_score = self._raw_popularity_scores(df)
        if "label" in df.columns:
            y_cls = df["label"].astype(np.int64).values
        else:
            y_cls = self._apply_thresholds(y_score)
        return y_reg, y_cls, y_score.astype(np.float32)

    @property
    def label_thresholds(self) -> dict[str, float]:
        return {"p50": self._p50, "p80": self._p80, "p95": self._p95}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Đổi tên cột alias về tên chuẩn nếu cần."""
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
        return compute_popularity_score(
            df["likes"].clip(lower=0).values,
            df["comments"].clip(lower=0).values,
            df["reposts"].clip(lower=0).values,
            df["views"].clip(lower=0).values,
        ).astype(np.float32)

    def _apply_thresholds(self, scores: np.ndarray) -> np.ndarray:
        labels = np.zeros(len(scores), dtype=np.int64)
        labels[scores >= self._p50] = 1
        labels[scores >= self._p80] = 2
        labels[scores >= self._p95] = 3
        return labels
