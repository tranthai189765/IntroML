"""
src/ranking/x_scorer.py
Ranking cho bộ dữ liệu X/Twitter thực.

Ranking per Posts:
  Score = ln(0.01·V + L + 5·C + 10·R + 1)
  (đã được tính sẵn trong dataset là cột score_*h)

  Labels (theo phân vị score_6h):
    Label 3 – Viral   : Top 5%
    Label 2 – Popular : Top 5–20%
    Label 1 – Medium  : Top 20–50%
    Label 0 – Low     : Bottom 50%

Ranking per Topic:
  Topic_Score_K = Σ_{i ∈ Topic_K} (Base_Score_i × W_tier_i)

  Tier weights:
    Label 3 (Viral)   : W = 5.0
    Label 2 (Popular) : W = 3.0
    Label 1 (Medium)  : W = 1.0
    Label 0 (Low)     : W = 0.0  ← bài rác bị loại hoàn toàn

  Phiên bản EARLY (0.5h): dùng score_0_5h + label_0_5h → feature hợp lệ (không leak)
  Phiên bản FINAL (6h) : dùng score_6h  + label      → chỉ dùng để báo cáo
"""

import numpy as np
import pandas as pd

# ── Tier weights ──────────────────────────────────────────────────────────────
TIER_WEIGHTS: dict[int, float] = {
    0: 0.0,   # Low/Flop — bị loại khỏi đóng góp topic
    1: 1.0,   # Medium   — giữ nguyên giá trị
    2: 3.0,   # Popular  — khuếch đại ×3
    3: 5.0,   # Viral    — gánh vác sức mạnh lan truyền ×5
}

LABEL_NAMES = {0: "Low", 1: "Medium", 2: "Popular", 3: "Viral"}


# ─────────────────────────────────────────────────────────────────────────────
# Topic Ranking — EARLY (dùng làm feature, không leak future data)
# ─────────────────────────────────────────────────────────────────────────────
def compute_early_topic_scores(
    df: pd.DataFrame,
    group_col: str = "lang",
) -> pd.DataFrame:
    """
    Topic score tính từ snapshot 0.5h (không dùng future data).

    Topic_Score_K = Σ (score_0_5h_i × W_label_0_5h_i)  for i ∈ Topic_K

    Trả về DataFrame có cột:
      {group_col}, topic_score_05h, topic_rank_05h,
      topic_n_posts, topic_mean_05h,
      topic_viral_05h, topic_popular_05h, topic_medium_05h, topic_low_05h
    """
    tmp = df.copy()
    tmp["_w"]  = tmp["label_0_5h"].map(TIER_WEIGHTS).fillna(0.0)
    tmp["_ws"] = tmp["score_0_5h"] * tmp["_w"]

    agg = tmp.groupby(group_col).agg(
        topic_score_05h     = ("_ws",          "sum"),
        topic_n_posts       = ("score_0_5h",   "count"),
        topic_mean_05h      = ("score_0_5h",   "mean"),
        topic_viral_05h     = ("label_0_5h",   lambda x: (x == 3).sum()),
        topic_popular_05h   = ("label_0_5h",   lambda x: (x == 2).sum()),
        topic_medium_05h    = ("label_0_5h",   lambda x: (x == 1).sum()),
        topic_low_05h       = ("label_0_5h",   lambda x: (x == 0).sum()),
    ).reset_index()

    agg = agg.sort_values("topic_score_05h", ascending=False).reset_index(drop=True)
    agg["topic_rank_05h"] = range(1, len(agg) + 1)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Topic Ranking — FINAL (chỉ dùng để báo cáo / visualization)
# ─────────────────────────────────────────────────────────────────────────────
def compute_final_topic_scores(
    df: pd.DataFrame,
    group_col: str = "lang",
) -> pd.DataFrame:
    """
    Topic score tính từ snapshot 6h — KHÔNG dùng làm feature (data leakage).
    Dùng để so sánh với early prediction và vẽ báo cáo.

    Topic_Score_K = Σ (score_final_i × W_label_i)  for i ∈ Topic_K
    """
    tmp = df.copy()
    tmp["_w"]  = tmp["label"].map(TIER_WEIGHTS).fillna(0.0)
    tmp["_ws"] = tmp["score_final"] * tmp["_w"]

    agg = tmp.groupby(group_col).agg(
        topic_score_final   = ("_ws",          "sum"),
        topic_n_posts       = ("score_final",  "count"),
        topic_mean_final    = ("score_final",  "mean"),
        topic_viral_count   = ("label",        lambda x: (x == 3).sum()),
        topic_popular_count = ("label",        lambda x: (x == 2).sum()),
        topic_medium_count  = ("label",        lambda x: (x == 1).sum()),
        topic_low_count     = ("label",        lambda x: (x == 0).sum()),
    ).reset_index()

    agg = agg.sort_values("topic_score_final", ascending=False).reset_index(drop=True)
    agg["topic_rank_final"] = range(1, len(agg) + 1)

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Topic Stats Tracker — incremental, no future leakage
# ─────────────────────────────────────────────────────────────────────────────
class TopicStatsTracker:
    """
    Thống kê topic online, TÁCH RIÊNG THEO TỪNG SNAPSHOT (obs) → topic context
    luôn 'đồng tuổi' với mốc đang quan sát (vd obs=2h so với mặt bằng topic @2h),
    thay vì đóng băng ở 0.5h.

    Trong prequential stream, tại mỗi (post, obs_suffix):
      - get_features(snapshot, group)            → stats của các post ĐÃ THẤY ở CÙNG mốc
      - update(snapshot, group, score, label)    → cập nhật SAU khi train xong bài i

    Leak-safe: get_features() gọi trước update(); chỉ gộp các post đã đi qua cùng
    mốc obs trước đó trong stream.
    """

    def __init__(self, tier_weights: dict[int, float] = TIER_WEIGHTS) -> None:
        self._w = tier_weights
        # (snapshot, group) → {"ws", "n", "sum", "viral", "popular", "medium", "low"}
        self._stats: dict[tuple[str, str], dict] = {}

    def _get(self, snapshot: str, group: str) -> dict:
        key = (snapshot, group)
        if key not in self._stats:
            self._stats[key] = {
                "ws": 0.0, "n": 0, "sum": 0.0,
                "viral": 0, "popular": 0, "medium": 0, "low": 0,
            }
        return self._stats[key]

    def get_features(self, snapshot: str, group: str) -> dict:
        """Topic features @snapshot dựa trên các post ĐÃ THẤY ở cùng mốc."""
        s = self._get(snapshot, group)
        n    = s["n"]
        mean = s["sum"] / n if n > 0 else 0.0

        # rank: số group có ws cao hơn TRONG CÙNG snapshot (1-indexed)
        cur_ws = s["ws"]
        rank = 1 + sum(1 for (snap, g), gs in self._stats.items()
                       if snap == snapshot and g != group and gs["ws"] > cur_ws)

        return {
            "feat_topic_score":          np.log1p(cur_ws),
            "feat_topic_rank":           float(rank),
            "feat_topic_n_posts":        float(n),
            "feat_topic_mean":           mean,
            "feat_topic_viral_ratio":    s["viral"]   / (n + 1),
            "feat_topic_popular_ratio":  s["popular"] / (n + 1),
        }

    def update(self, snapshot: str, group: str, score: float, label: int) -> None:
        """Cập nhật stats @snapshot SAU KHI train xong bài hiện tại."""
        s = self._get(snapshot, group)
        s["ws"]  += score * self._w.get(label, 0.0)
        s["n"]   += 1
        s["sum"] += score
        if   label == 3: s["viral"]   += 1
        elif label == 2: s["popular"] += 1
        elif label == 1: s["medium"]  += 1
        else:            s["low"]     += 1

    def to_dataframe(self, group_col: str = "lang", snapshot: str = "0_5h") -> pd.DataFrame:
        """Stats của một snapshot dưới dạng DataFrame (để báo cáo)."""
        rows = []
        for (snap, g), s in self._stats.items():
            if snap != snapshot:
                continue
            n = max(s["n"], 1)
            rows.append({
                group_col:        g,
                "topic_score":    s["ws"],
                "topic_n_posts":  s["n"],
                "topic_mean":     s["sum"] / n,
                "topic_viral":    s["viral"],
                "topic_popular":  s["popular"],
                "topic_medium":   s["medium"],
                "topic_low":      s["low"],
            })
        if not rows:
            return pd.DataFrame()
        agg = pd.DataFrame(rows).sort_values("topic_score", ascending=False).reset_index(drop=True)
        agg["topic_rank"] = range(1, len(agg) + 1)
        return agg


def print_topic_ranking(
    early_df: pd.DataFrame,
    final_df: pd.DataFrame,
    group_col: str = "lang",
    top_n: int = 15,
) -> None:
    """In bảng ranking topic ra terminal."""
    from tabulate import tabulate

    print("\n-- Topic Ranking (EARLY @0.5h — used as feature) ------------------")
    cols_e = [group_col, "topic_rank_05h", "topic_score_05h",
              "topic_n_posts", "topic_viral_05h", "topic_popular_05h"]
    print(tabulate(early_df[cols_e].head(top_n), headers="keys",
                   tablefmt="grid", floatfmt=".2f"))

    print("\n-- Topic Ranking (FINAL @6h — report only) -------------------------")
    cols_f = [group_col, "topic_rank_final", "topic_score_final",
              "topic_n_posts", "topic_viral_count", "topic_popular_count"]
    print(tabulate(final_df[cols_f].head(top_n), headers="keys",
                   tablefmt="grid", floatfmt=".2f"))
    print("-" * 66)
