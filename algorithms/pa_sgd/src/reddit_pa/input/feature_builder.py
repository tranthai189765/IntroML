"""
Reddit FeatureBuilder.

Vector z = [z_title(32); z_body(32); z_subreddit(16); z_meta(5); z_engagement(4)]

z_title     : TF-IDF SVD trên tiêu đề bài đăng (32 chiều)
z_body      : TF-IDF SVD trên nội dung bài đăng (32 chiều)
z_subreddit : TF-IDF SVD trên tên subreddit (16 chiều)
z_meta      : [is_text_post, flair_hash, hour_of_day, day_of_week, log1p(title_len)]
z_engagement: [log1p(score_t), log1p(num_comments_t), upvote_ratio_t, log1p(delta_t+1)]
              Chỉ thêm khi add_engagement=True (temporal learning).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ...utils.text_encoder import TextEncoder


class RedditFeatureBuilder:
    """
    Xây dựng feature matrix z cho từng Reddit post.

    Parameters
    ----------
    title_dim : int     chiều SVD cho title (mặc định 32)
    body_dim  : int     chiều SVD cho selftext (mặc định 32)
    subreddit_dim : int chiều SVD cho subreddit name (mặc định 16)
    add_engagement : bool
        Thêm engagement features cho temporal learning.
    delta_t_col : str
        Cột chứa khoảng cách thời gian (giờ) giữa 2 snapshot.
    scale : bool        StandardScaler
    """

    META_DIM       = 5
    ENGAGEMENT_DIM = 4

    def __init__(
        self,
        title_dim: int = 32,
        body_dim: int = 32,
        subreddit_dim: int = 16,
        add_engagement: bool = False,
        delta_t_col: str = "delta_t_hours",
        scale: bool = True,
    ):
        self.title_enc     = TextEncoder(backend="tfidf", text_dim=title_dim)
        self.body_enc      = TextEncoder(backend="tfidf", text_dim=body_dim)
        # min_df=1: subreddit names are short unique words, not full sentences
        self.subreddit_enc = TextEncoder(backend="tfidf", text_dim=subreddit_dim, min_df=1)
        self.add_engagement = add_engagement
        self.delta_t_col   = delta_t_col
        self.scale         = scale

        self.scaler    = StandardScaler()
        self._fitted   = False
        self.feature_dim: int = 0

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        z = self._build_raw(df, fit=True)
        if self.scale:
            z = self.scaler.fit_transform(z)
        self._fitted = True
        self.feature_dim = z.shape[1]
        return z.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        z = self._build_raw(df, fit=False)
        if self.scale:
            z = self.scaler.transform(z)
        return z.astype(np.float32)

    def transform_single(self, row: dict | pd.Series) -> np.ndarray:
        return self.transform(pd.DataFrame([row]))

    def _build_raw(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        parts: list[np.ndarray] = []

        titles = df["title"].fillna("").tolist() if "title" in df.columns else [""] * len(df)
        if fit:
            self.title_enc.fit(list(dict.fromkeys(titles)))
        parts.append(self.title_enc.encode(titles))

        bodies = df["selftext"].fillna("").tolist() if "selftext" in df.columns else [""] * len(df)
        if fit:
            self.body_enc.fit(list(dict.fromkeys(bodies)))
        parts.append(self.body_enc.encode(bodies))

        subs = df["subreddit"].fillna("").tolist() if "subreddit" in df.columns else [""] * len(df)
        if fit:
            self.subreddit_enc.fit(list(dict.fromkeys(subs)))
        parts.append(self.subreddit_enc.encode(subs))

        parts.append(self._extract_meta(df))

        if self.add_engagement:
            parts.append(self._extract_engagement(df))

        return np.hstack(parts)

    def _extract_meta(self, df: pd.DataFrame) -> np.ndarray:
        """
        5-dim meta vector:
          [is_text_post, flair_hash/100, hour_of_day/23, day_of_week/6, log1p(title_len)]
        """
        n = len(df)
        meta = np.zeros((n, self.META_DIM), dtype=np.float32)

        # is_text_post: 1 nếu có selftext
        if "selftext" in df.columns:
            meta[:, 0] = (df["selftext"].fillna("") != "").astype(float).values

        # flair: hash tên flair về [0,1]
        if "link_flair_text" in df.columns:
            flair_hash = df["link_flair_text"].fillna("").apply(
                lambda x: hash(x) % 100 / 100.0
            ).values.astype(np.float32)
            meta[:, 1] = flair_hash

        # hour_of_day
        if "created_utc" in df.columns:
            dt = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
            meta[:, 2] = dt.dt.hour.fillna(0).values / 23.0

        # day_of_week
        if "created_utc" in df.columns:
            dt = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
            meta[:, 3] = dt.dt.dayofweek.fillna(0).values / 6.0

        # log1p(title_len)
        if "title" in df.columns:
            lens = df["title"].fillna("").str.len().values.astype(np.float32)
            meta[:, 4] = np.log1p(lens)

        return meta

    def _extract_engagement(self, df: pd.DataFrame) -> np.ndarray:
        """
        4-dim engagement vector cho temporal learning:
          [log1p(score_t), log1p(num_comments_t), upvote_ratio_t, log1p(delta_t+1)]
        """
        n = len(df)
        z = np.zeros((n, self.ENGAGEMENT_DIM), dtype=np.float32)

        if "score" in df.columns:
            vals = pd.to_numeric(df["score"], errors="coerce").fillna(0).clip(lower=0).values
            z[:, 0] = np.log1p(vals)

        if "num_comments" in df.columns:
            vals = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0).clip(lower=0).values
            z[:, 1] = np.log1p(vals)

        if "upvote_ratio" in df.columns:
            vals = pd.to_numeric(df["upvote_ratio"], errors="coerce").fillna(0.5).clip(0, 1).values
            z[:, 2] = vals.astype(np.float32)

        if self.delta_t_col in df.columns:
            dt = pd.to_numeric(df[self.delta_t_col], errors="coerce").fillna(0).clip(lower=0).values
            z[:, 3] = np.log1p(dt + 1)

        return z

    @property
    def engagement_dim(self) -> int:
        return self.ENGAGEMENT_DIM if self.add_engagement else 0
