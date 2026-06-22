"""
src/features/x_extractor.py
Feature engineering từ snapshot 0.5h của bộ dữ liệu X/Twitter.

Hai nguồn feature TÁCH BẠCH:
  • build_stream_instance() — feature THỰC SỰ train cho ARF (multi-snapshot,
    obs→next): engagement log, velocity log, ratios, growth-since-baseline,
    author/content tĩnh, image PCA, và topic context per-snapshot (từ tracker).
  • extract_x_features() — dựng các cột df cố-định @0.5h (author/content/image +
    snapshot 0.5h) phục vụ báo cáo (proxy feature-importance), KHÔNG phải input model.

Topic context KHÔNG nằm trong df — được cấp per-snapshot qua TopicStatsTracker
ngay trong build_stream_instance (leak-safe, đồng tuổi với mốc dự đoán).
"""

import numpy as np
import pandas as pd

X_FEATURE_COLS: list[str] = []   # điền sau khi extract_x_features chạy

# (obs_suffix, obs_h, next_suffix, next_h) — next-step prediction pairs
# obs_h / next_h dùng làm feature để model biết đang train/predict ở mốc nào
SNAPSHOT_PAIRS: list[tuple[str, float, str, float]] = [
    ("0_5h", 0.5, "1h",   1.0),
    ("1h",   1.0, "1_5h", 1.5),
    ("1_5h", 1.5, "2h",   2.0),
    ("2h",   2.0, "3h",   3.0),
    ("3h",   3.0, "4h",   4.0),
    ("4h",   4.0, "6h",   6.0),
    ("6h",   6.0, "10h",  10.0),
    ("10h",  10.0, "16h", 16.0),
    ("16h",  16.0, "24h", 24.0),
    ("24h",  24.0, "48h", 48.0),
    ("48h",  48.0, "60h", 60.0),
    ("60h",  60.0, "72h", 72.0),
]

# Snapshot sớm nhất = baseline cho các feature growth-since-baseline.
# Lấy động từ SNAPSHOT_PAIRS để không hardcode "0_5h".
BASELINE_SUFFIX: str   = SNAPSHOT_PAIRS[0][0]
BASELINE_H:      float = SNAPSHOT_PAIRS[0][1]

# Cột tĩnh (không thay đổi theo snapshot) — đọc trực tiếp từ hàng df đã qua extract_x_features
_STATIC_FEAT_COLS = [
    "feat_log_followers", "feat_blue_verified", "feat_legacy_verified",
    "feat_log_fol_per_day", "feat_log_ff_ratio", "feat_log_author_age",
    "feat_author_found", "feat_has_image", "feat_has_video",
    "feat_log_text_len", "feat_intake_age_h",
]


def build_stream_instance(
    row: pd.Series,
    snap_suffix: str,
    obs_h: float,
    next_h: float,
    topic_features: dict | None = None,
) -> dict:
    """
    Xây dựng feature dict cho một (post, snapshot) trong multi-snapshot stream.

    - Engagement features tính từ cột `*_{snap_suffix}` của row
    - Static features (author, content, image) đọc từ cột feat_* đã pre-extract
    - Topic features nhận từ TopicStatsTracker (nếu có)
    - Time features: feat_obs_h, feat_next_h, feat_delta_h
    """
    v = float(row.get(f"views_{snap_suffix}",    0) or 0)
    l = float(row.get(f"likes_{snap_suffix}",    0) or 0)
    c = float(row.get(f"comments_{snap_suffix}", 0) or 0)
    r = float(row.get(f"reposts_{snap_suffix}",  0) or 0)
    score = float(np.log(0.01 * v + l + 5 * c + 10 * r + 1))

    # ── Baseline (snapshot sớm nhất) cho growth-since-baseline ─────────────────
    # baseline_h ≤ obs_h luôn đúng → growth không leak tương lai.
    bv = float(row.get(f"views_{BASELINE_SUFFIX}",    0) or 0)
    bl = float(row.get(f"likes_{BASELINE_SUFFIX}",    0) or 0)
    bc = float(row.get(f"comments_{BASELINE_SUFFIX}", 0) or 0)
    br = float(row.get(f"reposts_{BASELINE_SUFFIX}",  0) or 0)
    base_score = float(np.log(0.01 * bv + bl + 5 * bc + 10 * br + 1))

    feats: dict = {
        # ── Time features — model biết đang quan sát lúc nào và predict đến đâu
        "feat_obs_h":               obs_h,
        "feat_next_h":              next_h,
        "feat_delta_h":             next_h - obs_h,
        # ── Engagement counts (chỉ log) ────────────────────────────────────────
        # ARF = rừng cây Hoeffding → split theo ngưỡng, BẤT BIẾN với biến đổi đơn
        # điệu. Giữ cả raw lẫn log là dư thừa (cho cùng split) và còn làm loãng
        # không gian con max_features='sqrt' của mỗi cây. Chỉ giữ log: ổn định số
        # học + giúp bộ ước lượng điểm split của river chọn ngưỡng tốt trên đuôi nặng.
        "feat_log_views":           float(np.log1p(v)),
        "feat_log_likes":           float(np.log1p(l)),
        "feat_log_comments":        float(np.log1p(c)),
        "feat_log_reposts":         float(np.log1p(r)),
        "feat_score":               score,
        # ── Velocity (log đơn vị/giờ) — log-hóa vì phân phối đuôi nặng ─────────
        "feat_log_views_vel":       float(np.log1p(v / obs_h)),
        "feat_log_likes_vel":       float(np.log1p(l / obs_h)),
        "feat_log_comments_vel":    float(np.log1p(c / obs_h)),
        "feat_log_reposts_vel":     float(np.log1p(r / obs_h)),
        # ── Ratios ────────────────────────────────────────────────────────────
        "feat_likes_to_views":      l / max(v, 1),
        "feat_reposts_to_views":    r / max(v, 1),
        "feat_comments_to_views":   c / max(v, 1),
        "feat_reposts_to_likes":    r / (l + 1),
        "feat_comments_to_likes":   c / (l + 1),
        "feat_log_total_action":    float(np.log1p(l + c + r)),
        # ── Growth-since-baseline (log-diff so với snapshot sớm nhất) ──────────
        # = 0 khi đang ở chính baseline; >0 khi bài tăng trưởng từ baseline → obs
        "feat_growth_log_views":    float(np.log1p(v) - np.log1p(bv)),
        "feat_growth_log_likes":    float(np.log1p(l) - np.log1p(bl)),
        "feat_growth_log_comments": float(np.log1p(c) - np.log1p(bc)),
        "feat_growth_log_reposts":  float(np.log1p(r) - np.log1p(br)),
        "feat_growth_score":        score - base_score,
    }

    # ── Static author / content features (pre-computed) ──────────────────────
    for col in _STATIC_FEAT_COLS:
        feats[col] = float(row.get(col, 0) or 0)

    # ── Embedding pre-computed (image/text/author PCA) ────────────────────────
    for col in row.index:
        if col.startswith("feat_img_") or col.startswith("feat_emb_"):
            feats[col] = float(row[col] or 0)

    # ── Topic context từ TopicStatsTracker ────────────────────────────────────
    if topic_features:
        feats.update(topic_features)
        # post_vs_topic: điểm snapshot hiện tại so với mặt bằng topic CÙNG mốc.
        # Cold-start (chưa có post nào ở mốc này) → 1.0 trung tính, tránh blow-up.
        topic_mean = topic_features.get("feat_topic_mean", 0.0)
        feats["feat_post_vs_topic"] = score / topic_mean if topic_mean > 1e-9 else 1.0

    return feats


def extract_x_features(
    df: pd.DataFrame,
    img_emb_df: pd.DataFrame | None = None,
    group_col: str = "lang",
) -> pd.DataFrame:
    """
    Dựng các cột df @0.5h phục vụ BÁO CÁO (proxy feature-importance) + các cột
    tĩnh (author/content/image) mà build_stream_instance đọc lại theo từng row.
    Cuối hàm đăng ký X_FEATURE_COLS = đúng feature mà build_stream_instance sinh ra.
    """
    df = df.copy()

    # ── A. Early Engagement Counts (raw + log) ────────────────────────────────
    for m in ["views", "likes", "comments", "reposts"]:
        col = f"{m}_0_5h"
        df[f"feat_{m}_05h"]     = df[col]
        df[f"feat_log_{m}_05h"] = np.log1p(df[col])

    df["feat_score_05h"]     = df["score_0_5h"]

    # ── B. Velocity (per hour — chia 0.5 để ra đơn vị/giờ) ──────────────────
    for m in ["views", "likes", "comments", "reposts"]:
        df[f"feat_{m}_vel"] = df[f"{m}_0_5h"] / 0.5

    # ── C. Engagement Ratios ──────────────────────────────────────────────────
    v = df["views_0_5h"].replace(0, np.nan)
    l = df["likes_0_5h"]
    c = df["comments_0_5h"]
    r = df["reposts_0_5h"]

    df["feat_likes_to_views"]    = (l / v).fillna(0)
    df["feat_reposts_to_views"]  = (r / v).fillna(0)
    df["feat_comments_to_views"] = (c / v).fillna(0)
    df["feat_reposts_to_likes"]  = (r / (l + 1))
    df["feat_comments_to_likes"] = (c / (l + 1))

    # Total action (likes + comments + reposts — viral thường có reposts >> likes)
    df["feat_total_action_05h"]  = l + c + r

    # ── D. Score early ────────────────────────────────────────────────────────
    # score_0_5h đã log-scale → dùng trực tiếp (tránh log log)

    # ── E. Author Signals ─────────────────────────────────────────────────────
    df["feat_log_followers"]     = df["author_log_followers"]
    df["feat_blue_verified"]     = df["author_blue_verified"].astype(int)
    df["feat_legacy_verified"]   = df["author_verified"].astype(int)
    df["feat_log_fol_per_day"]   = np.log1p(df["author_followers_per_day"])
    df["feat_log_ff_ratio"]      = np.log1p(df["author_ff_ratio"])
    df["feat_log_author_age"]    = np.log1p(df["author_age_days"])
    df["feat_author_found"]      = df["author_found"].astype(int)

    # ── F. Content Metadata ───────────────────────────────────────────────────
    df["feat_has_image"]         = df["has_image"].astype(int)
    df["feat_has_video"]         = df["has_video"].astype(int)
    df["feat_log_text_len"]      = np.log1p(df["text_len"])

    # ── G. Intake Age ─────────────────────────────────────────────────────────
    df["feat_intake_age_h"]      = df["intake_age_h"]

    # ── H. Topic Context ──────────────────────────────────────────────────────
    # KHÔNG dựng ở df: topic context được cấp per-snapshot qua TopicStatsTracker
    # ngay trong build_stream_instance (leak-safe, đồng tuổi với mốc dự đoán).

    # ── I. Image Embedding (ResNet18 → PCA) ───────────────────────────────────
    if img_emb_df is not None:
        # Merge theo id; bài không có ảnh đã là 0 trong img_emb_df
        img_cols = [c for c in img_emb_df.columns if c.startswith("feat_img_")]
        df = df.merge(img_emb_df[["id"] + img_cols], on="id", how="left")
        for col in img_cols:
            df[col] = df[col].fillna(0.0)

    # ── Đăng ký X_FEATURE_COLS = đúng các key mà build_stream_instance sinh ra ──
    # Nguồn chân lý duy nhất: chỉ feature THỰC SỰ được train mới có mặt ở đây →
    # các cột df @0.5h (feat_*_05h) chỉ phục vụ báo cáo, không lọt vào feature set.
    from src.ranking.x_scorer import TopicStatsTracker

    sample_row    = df.iloc[0]
    sample_topic  = TopicStatsTracker().get_features(BASELINE_SUFFIX, sample_row[group_col])
    sample_instance = build_stream_instance(
        sample_row, BASELINE_SUFFIX, BASELINE_H, SNAPSHOT_PAIRS[0][3], sample_topic
    )
    X_FEATURE_COLS.clear()
    X_FEATURE_COLS.extend(sample_instance.keys())

    return df


def get_x_feature_dict(row: pd.Series) -> dict:
    return {col: float(row[col]) for col in X_FEATURE_COLS if col in row.index}
