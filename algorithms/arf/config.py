"""
config.py - Cấu hình toàn bộ dự án
Trending Topic & Post Prediction using Adaptive Random Forest (ARF)
"""
import os

# ── Topics (chủ đề bài viết) ────────────────────────────────────────────────
TOPICS = [
    "Sports",
    "Machine_Learning",
    "Technology",
    "Politics",
    "Entertainment",
    "Science",
    "Health",
    "Business",
    "Gaming",
    "Travel",
]
N_TOPICS = len(TOPICS)

# ── Dataset ──────────────────────────────────────────────────────────────────
N_POSTS     = 10_000   # số bài viết trong 1 snapshot
RANDOM_SEED = 42

# ── Snapshot (giờ so với hiện tại) ───────────────────────────────────────────
SNAPSHOT_1H  = 1    # snapshot 1 giờ sau khi đăng  → dùng làm INPUT (features)
SNAPSHOT_24H = 24   # snapshot 24 giờ sau khi đăng → dùng làm TARGET (labels)

# ── Trọng số tính Engagement Score ──────────────────────────────────────────
W_LIKES    = 1.0   # likes
W_SHARES   = 2.0   # shares lan rộng nội dung → trọng số cao hơn
W_COMMENTS = 1.5   # comments thể hiện tương tác tích cực

# ── Hacker News Hot Score – tham số gravity (time-decay) ────────────────────
HOT_GRAVITY = 1.8

# ── Ngưỡng phân loại "popular" ───────────────────────────────────────────────
TOP_K_TOPICS    = 3     # top 3/10 topics → "popular topic"
TOP_FRAC_POSTS  = 0.10  # top 10% bài viết trong topic → "popular post"

# ── Adaptive Random Forest ───────────────────────────────────────────────────
ARF_N_MODELS      = 10    # số cây trong rừng (10 = nhanh; 100 = chính xác hơn)
ARF_LAMBDA        = 6     # Poisson lambda cho online bagging
ARF_GRACE_PERIOD  = 50    # số mẫu giữa các lần thử split
ARF_MAX_DEPTH     = None  # không giới hạn độ sâu
ARF_SEED          = RANDOM_SEED

# ── Online Evaluation ────────────────────────────────────────────────────────
WARM_UP_SIZE = 500   # số mẫu khởi động trước khi đánh giá
EVAL_WINDOW  = 200   # cửa sổ trượt cho prequential evaluation

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")
