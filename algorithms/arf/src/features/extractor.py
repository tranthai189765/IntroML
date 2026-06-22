"""
src/features/extractor.py
Feature Engineering: Trích xuất tập đặc trưng từ snapshot 1h cho mô hình ARF.

Nhóm features:
  A. Engagement Counts     – likes, shares, comments tại 1h
  B. Engagement Velocity   – tốc độ tích lũy engagement (per hour)
  C. Engagement Ratios     – tỉ lệ giữa các loại tương tác
  D. Ranking Scores        – 5 loại score đã tính ở scorer.py
  E. Topic Context         – thống kê topic tại 1h
  F. Content Metadata      – word_count, has_image, has_link
  G. Temporal              – post_hour, post_dow
"""

import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import TOPICS


FEATURE_COLS: list[str] = []  # sẽ được điền sau khi extract_features chạy


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các cột feature mới (tiền tố 'feat_') từ DataFrame đã qua preprocessor.
    Trả về DataFrame với các cột feat_* sẵn sàng cho river's ARF.
    """
    df = df.copy()

    # ── A. Engagement Counts ──────────────────────────────────────────────────
    df["feat_likes_1h"]    = df["likes_1h"]
    df["feat_shares_1h"]   = df["shares_1h"]
    df["feat_comments_1h"] = df["comments_1h"]
    df["feat_total_eng_1h"] = (
        df["likes_1h"] + df["shares_1h"] + df["comments_1h"]
    )

    # ── B. Velocity (per hour, snapshot = 1h) ─────────────────────────────────
    df["feat_likes_velocity"]    = df["likes_1h"]    / 1.0   # likes/hr at 1h
    df["feat_shares_velocity"]   = df["shares_1h"]   / 1.0
    df["feat_comments_velocity"] = df["comments_1h"] / 1.0

    # ── C. Engagement Ratios ──────────────────────────────────────────────────
    total     = df["feat_total_eng_1h"].replace(0, np.nan)
    df["feat_likes_ratio"]    = (df["likes_1h"]    / total).fillna(0)
    df["feat_shares_ratio"]   = (df["shares_1h"]   / total).fillna(0)
    df["feat_comments_ratio"] = (df["comments_1h"] / total).fillna(0)

    # shares-to-likes ratio: bài viral thường có shares >> likes
    df["feat_shares_likes_ratio"] = (
        df["shares_1h"] / (df["likes_1h"] + 1)
    )

    # ── D. Ranking Scores (đã tính trong scorer.py) ───────────────────────────
    df["feat_hot_score"]      = df["hot_score_1h"]
    df["feat_reddit_score"]   = df["reddit_score_1h"]
    df["feat_weighted_score"] = df["weighted_score_1h"]
    df["feat_velocity_score"] = df["velocity_score_1h"]
    df["feat_wilson_score"]   = df["wilson_score_1h"]

    # ── E. Topic Context ──────────────────────────────────────────────────────
    df["feat_topic_total_score"]   = df["topic_total_eng_1h"]
    df["feat_topic_mean_score"]    = df["topic_mean_eng_1h"]
    df["feat_topic_median_score"]  = df["topic_median_eng_1h"]
    df["feat_topic_n_posts"]       = df["topic_n_posts"]
    # Mức độ nổi bật của post so với topic trung bình
    df["feat_post_vs_topic_ratio"] = df["post_vs_topic_ratio"]

    # ── F. Content Metadata ───────────────────────────────────────────────────
    df["feat_word_count"]  = df["word_count"]
    df["feat_has_image"]   = df["has_image"]
    df["feat_has_link"]    = df["has_link"]

    # log-transform word count để giảm skew
    df["feat_log_word_count"] = np.log1p(df["word_count"])

    # ── G. Temporal Features ──────────────────────────────────────────────────
    df["feat_post_hour"] = df["post_hour"]
    df["feat_post_dow"]  = df["post_dow"]

    # Peak-hour indicator (7-10h, 12-14h, 19-22h)
    df["feat_is_peak_hour"] = df["post_hour"].apply(
        lambda h: int((7 <= h <= 10) or (12 <= h <= 14) or (19 <= h <= 22))
    )

    # ── H. Topic One-Hot (cần thiết nếu ARF không handle string) ─────────────
    for tid, tname in enumerate(TOPICS):
        df[f"feat_topic_{tname}"] = (df["topic"] == tname).astype(int)

    # Mutate FEATURE_COLS in-place (giữ reference hợp lệ cho các module khác)
    FEATURE_COLS.clear()
    FEATURE_COLS.extend([c for c in df.columns if c.startswith("feat_")])

    return df


def get_feature_dict(row: pd.Series) -> dict:
    """Chuyển một dòng DataFrame thành dict phù hợp với river's learn_one/predict_one."""
    return {col: float(row[col]) for col in FEATURE_COLS if col in row.index}
