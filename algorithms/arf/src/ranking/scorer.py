"""
src/ranking/scorer.py
Survey & Implementation của 5 thuật toán Ranking phổ biến.

Các thuật toán được khảo sát:
  1. Weighted Sum Score      – baseline đơn giản
  2. Reddit Hot Score        – logarithmic + timestamp offset
  3. Hacker News Score       – time-decay gravity (ĐỀ XUẤT SỬ DỤNG)
  4. Engagement Velocity     – tỉ lệ tăng trưởng theo thời gian
  5. Wilson Score Lower Bound – giới hạn dưới theo thống kê Bayesian
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import W_LIKES, W_SHARES, W_COMMENTS, HOT_GRAVITY


# ─────────────────────────────────────────────────────────────────────────────
# 1. Weighted Sum Score (Baseline)
# ─────────────────────────────────────────────────────────────────────────────
def weighted_sum_score(likes: float, shares: float, comments: float,
                       w_l: float = W_LIKES,
                       w_s: float = W_SHARES,
                       w_c: float = W_COMMENTS) -> float:
    """
    Score = w_l * likes + w_s * shares + w_c * comments

    Ưu điểm : Đơn giản, dễ giải thích.
    Nhược điểm: Không tính đến yếu tố thời gian; bài cũ vẫn giữ score cao.
    """
    return w_l * likes + w_s * shares + w_c * comments


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reddit Hot Score
# ─────────────────────────────────────────────────────────────────────────────
def reddit_hot_score(likes, shares, comments,
                     post_age_hours: float,
                     epoch_seconds: float = 1_134_028_003.0):
    """
    Phỏng theo thuật toán xếp hạng của Reddit (Randall Munroe, 2009).

    score = sign(s) * log10(max(|s|, 1)) + t / 45000
    trong đó s = engagement, t = thời điểm đăng (unix seconds).

    Bài đăng MỚI hơn được cộng thêm điểm thời gian t/45000.
    Ưu điểm : Cân bằng giữa engagement và độ mới.
    Nhược điểm: Không giảm điểm liên tục theo tuổi bài viết.
    """
    s     = W_LIKES * likes + W_SHARES * shares + W_COMMENTS * comments
    sign  = np.sign(s)
    order = np.log10(np.maximum(np.abs(s), 1))
    t     = epoch_seconds - post_age_hours * 3600
    return sign * order + t / 45_000.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hacker News Hot Score  ← THUẬT TOÁN ĐỀ XUẤT
# ─────────────────────────────────────────────────────────────────────────────
def hacker_news_score(likes: float, shares: float, comments: float,
                      post_age_hours: float,
                      gravity: float = HOT_GRAVITY) -> float:
    """
    Phỏng theo thuật toán xếp hạng của Hacker News (Paul Graham).

    score = P / (T + 2)^G
    trong đó:
      P = weighted engagement (likes + 2*shares + 1.5*comments)
      T = tuổi bài viết tính bằng giờ
      G = gravity (default 1.8) — độ suy giảm điểm theo thời gian

    Ưu điểm : Bài mới có lợi thế rõ rệt; G lớn → độ tươi quan trọng hơn.
    Nhược điểm: Bài cũ nhưng viral vẫn có thể bị đẩy xuống thấp.

    >>> round(hacker_news_score(100, 50, 30, age_hours=2), 2)
    """
    P = W_LIKES * likes + W_SHARES * shares + W_COMMENTS * comments
    T = max(0, post_age_hours)
    return P / (T + 2) ** gravity


# ─────────────────────────────────────────────────────────────────────────────
# 4. Engagement Velocity Score
# ─────────────────────────────────────────────────────────────────────────────
def engagement_velocity_score(likes: float, shares: float, comments: float,
                               post_age_hours: float) -> float:
    """
    score = total_engagement / sqrt(age_hours + 1)

    Tập trung vào TỐC ĐỘ tích lũy engagement.
    Ưu điểm : Phát hiện bài đang "nổi lên" nhanh.
    Nhược điểm: Bài rất mới (age → 0) bị khuếch đại quá mức.
    """
    P = W_LIKES * likes + W_SHARES * shares + W_COMMENTS * comments
    return P / np.sqrt(max(post_age_hours, 0.1) + 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Wilson Score Lower Bound
# ─────────────────────────────────────────────────────────────────────────────
def wilson_lower_bound(positive: float, total: float,
                       z: float = 1.96) -> float:
    """
    Giới hạn dưới của khoảng tin cậy Wilson (z=1.96 ↔ 95% CI).

    score = (p̂ + z²/2n - z*sqrt(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)
    trong đó p̂ = positive / total.

    Ưu điểm : Ổn định khi số lượng tương tác nhỏ; tính không chắc chắn vào điểm.
    Nhược điểm: Chỉ phù hợp cho phản hồi nhị phân (like/dislike).
    Áp dụng: total = likes + dislikes (proxy: comments = "dislike engagement")
    """
    if total == 0:
        return 0.0
    p_hat = positive / total
    n     = total
    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Áp dụng tất cả scoring lên DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_scores(df: pd.DataFrame,
                       snapshot: str = "1h") -> pd.DataFrame:
    """
    Tính 5 loại ranking score dựa trên cột likes_{snapshot},
    shares_{snapshot}, comments_{snapshot}.

    Bổ sung thêm:
      - weighted_score_{snapshot}   (Weighted Sum)
      - reddit_score_{snapshot}     (Reddit Hot)
      - hot_score_{snapshot}        (Hacker News) ← CHỦ YẾU
      - velocity_score_{snapshot}   (Velocity)
      - wilson_score_{snapshot}     (Wilson Lower Bound)
    """
    l = df[f"likes_{snapshot}"].values.astype(float)
    s = df[f"shares_{snapshot}"].values.astype(float)
    c = df[f"comments_{snapshot}"].values.astype(float)

    # Tuổi bài: với 1h snapshot → T = 1h; với 24h snapshot → T = 24h
    age_h = float(snapshot.replace("h", ""))

    df[f"weighted_score_{snapshot}"] = weighted_sum_score(l, s, c)
    df[f"reddit_score_{snapshot}"]   = reddit_hot_score(l, s, c, age_h)
    df[f"hot_score_{snapshot}"]      = hacker_news_score(l, s, c, age_h)
    df[f"velocity_score_{snapshot}"] = engagement_velocity_score(l, s, c, age_h)

    total  = l + c        # likes = positive, comments = engagement total proxy
    df[f"wilson_score_{snapshot}"] = [
        wilson_lower_bound(float(li), float(li + ci))
        for li, ci in zip(l, c)
    ]

    return df


def rank_topics(df: pd.DataFrame, score_col: str = "hot_score_1h") -> pd.DataFrame:
    """
    Tính tổng score theo topic → trả về bảng xếp hạng topic (Table 1).
    """
    topic_rank = (
        df.groupby("topic")[score_col]
        .agg(["sum", "mean", "count"])
        .rename(columns={"sum": "total_score", "mean": "avg_score", "count": "n_posts"})
        .sort_values("total_score", ascending=False)
        .reset_index()
    )
    topic_rank["rank"] = range(1, len(topic_rank) + 1)
    return topic_rank


def rank_posts_per_topic(df: pd.DataFrame,
                         score_col: str = "hot_score_1h") -> pd.DataFrame:
    """
    Xếp hạng bài viết trong từng topic theo score → Table 2.
    Thêm cột 'intra_topic_rank' (hạng trong topic) và 'intra_topic_rank_pct'.
    """
    df = df.copy()
    df["intra_topic_rank"] = df.groupby("topic")[score_col].rank(
        ascending=False, method="first"
    )
    df["topic_size"] = df.groupby("topic")["post_id"].transform("count")
    df["intra_topic_rank_pct"] = df["intra_topic_rank"] / df["topic_size"]
    return df
