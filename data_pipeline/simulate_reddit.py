# -*- coding: utf-8 -*-
"""
Mô phỏng dữ liệu Reddit (SYNTHETIC) từ động học tương tác của Twitter.
Reddit chỉ có 3 trường quan sát: upvote, downvote, views.

Ý tưởng:
  - Mỗi post Twitter cung cấp (a) hình dạng tăng-bão hòa theo thời gian và
    (b) độ lớn tương tác (likes/reposts/views) -> dùng làm "động cơ".
  - Ánh xạ sang Reddit:
      upvotes_t  ~ (likes_t + 0.5*reposts_t) * U_post * noise   (đơn điệu không giảm)
      ratio ρ    : phần lớn 0.80-0.95, ~12% post "gây tranh cãi" ρ~0.5
      downvotes_t = upvotes_t * (1-ρ)/ρ
      views_t    : từ views Twitter (scale) NHƯNG >= tổng vote * hệ số lurker
  - score = ln(0.01*views + (upvotes - downvotes) + 1)   (tương tự viral score Twitter)
  - label : phân vị Low50 / Medium30 / Popular15 / Viral5 trên score từng mốc.

KHÔNG phải dữ liệu thật. Chỉ để minh hoạ pipeline đa nền tảng.
"""
import os
import numpy as np
import pandas as pd

SRC = os.environ.get("SRC_CSV", "data_v1/dataset_72h.csv")
OUT = os.environ.get("OUT_CSV", "data_v1/dataset_reddit_72h.csv")
SEED = int(os.environ.get("SEED", "2026"))
GRID = [0.5, 1, 1.5, 2, 3, 4, 6, 10, 16, 24, 48, 60, 72]

# phân vị nhãn (giống Twitter): Low 50% / Medium 30% / Popular 15% / Viral 5%
Q = [0.50, 0.80, 0.95]


def gtag(g):
    return str(g).replace(".", "_")


def series(df, metric):
    cols = [f"{metric}_{gtag(g)}h" for g in GRID]
    return df[cols].to_numpy(dtype=float)  # (n, T)


def pct_labels(score_col):
    """digitize theo phân vị trong-mốc -> 0..3."""
    thr = np.quantile(score_col, Q)
    # đảm bảo ngưỡng tăng nghiêm ngặt (tránh trùng khi nhiều giá trị 0)
    for i in range(1, len(thr)):
        if thr[i] <= thr[i - 1]:
            thr[i] = np.nextafter(thr[i - 1], np.inf)
    return np.digitize(score_col, thr).astype(int)


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(SRC)
    n = len(df)
    print(f"[src] {n} Twitter posts -> mô phỏng {n} Reddit posts")

    likes = series(df, "likes")
    reposts = series(df, "reposts")
    views_tw = series(df, "views")
    T = len(GRID)

    # xáo trộn để post Reddit không trùng dòng với Twitter (trông độc lập)
    perm = rng.permutation(n)
    likes, reposts, views_tw = likes[perm], reposts[perm], views_tw[perm]

    # ---- tham số mức post ----
    U = rng.lognormal(mean=np.log(2.2), sigma=0.45, size=n)        # bội số upvote
    V = rng.lognormal(mean=0.0, sigma=0.50, size=n)                # bội số views
    lurk = rng.uniform(25, 130, size=n)                           # views / tổng vote tối thiểu

    # upvote_ratio: hỗn hợp "bình thường" vs "gây tranh cãi"
    controversial = rng.random(n) < 0.12
    ratio = np.where(
        controversial,
        rng.beta(3.0, 3.0, size=n) * 0.4 + 0.40,   # ~0.40-0.80, tâm 0.6
        rng.beta(12.0, 2.2, size=n) * 0.18 + 0.80,  # ~0.80-0.98, tâm ~0.89
    )
    ratio = np.clip(ratio, 0.45, 0.985)

    # ---- sinh chuỗi thời gian ----
    pos = likes + 0.5 * reposts                      # "động cơ" tích cực
    snap_noise = rng.lognormal(0.0, 0.05, size=(n, T))
    upv = (pos + 1.0) * U[:, None] * snap_noise
    upv = np.maximum.accumulate(upv, axis=1)         # đơn điệu không giảm
    upv = np.rint(upv).astype(np.int64)

    downv = np.rint(upv * (1 - ratio[:, None]) / ratio[:, None]).astype(np.int64)
    total_votes = upv + downv

    # views: hình dạng = pha trộn (đầu nhanh theo upvote) + (đuôi mượt theo views
    # Twitter, vốn chưa chạm 100% tại 24/48h) -> giữ "gần gần 100%" thay vì 100%.
    W = 0.65
    Fup = upv[:, -1].astype(float); Fup[Fup <= 0] = 1.0
    up_norm = np.clip(upv / Fup[:, None], 0.0, 1.0)
    Ftw = views_tw[:, -1].astype(float); Ftw[Ftw <= 0] = 1.0
    vtw_norm = np.clip(views_tw / Ftw[:, None], 0.0, 1.0)
    shape = W * up_norm + (1.0 - W) * vtw_norm        # = 1.0 đúng tại mốc 72h
    F_views = np.maximum(views_tw[:, -1] * V, total_votes[:, -1] * lurk)  # độ lớn tại 72h
    views = np.maximum(F_views[:, None] * shape, total_votes)             # views >= tổng vote
    views = np.maximum.accumulate(views, axis=1)
    views = np.rint(views).astype(np.int64)

    net = upv - downv
    score = np.log(0.01 * views + net + 1.0)
    score = np.where(np.isfinite(score), score, 0.0)

    # ---- đóng gói DataFrame ----
    out = pd.DataFrame()
    out["post_id"] = [f"reddit_{i:06d}" for i in range(n)]
    out["src_post_id"] = df["id"].to_numpy()[perm]
    out["upvote_ratio"] = np.round(ratio, 4)
    out["controversial"] = controversial.astype(int)

    for j, g in enumerate(GRID):
        t = gtag(g)
        out[f"upvotes_{t}h"] = upv[:, j]
        out[f"downvotes_{t}h"] = downv[:, j]
        out[f"views_{t}h"] = views[:, j]
        out[f"score_{t}h"] = np.round(score[:, j], 4)
        out[f"label_{t}h"] = pct_labels(score[:, j])

    out["score_final"] = np.round(score[:, -1], 4)
    out["label"] = pct_labels(score[:, -1])
    out.to_csv(OUT, index=False)

    # ---- tóm tắt ----
    print(f"[out] {OUT}  ({out.shape[0]} rows x {out.shape[1]} cols)")
    print("\n== Phân phối nhãn cuối (72h) ==")
    print(out["label"].value_counts().sort_index().to_string())
    print("\n== upvote_ratio ==")
    print(f"  median={np.median(ratio):.3f}  mean={ratio.mean():.3f}  "
          f"%controversial(ratio<0.8)={(ratio < 0.8).mean()*100:.1f}%")
    print("\n== thống kê mốc 72h ==")
    for c in ["upvotes_72h", "downvotes_72h", "views_72h"]:
        s = out[c]
        print(f"  {c:14}: median={int(s.median()):>8}  p90={int(s.quantile(.9)):>9}  "
              f"max={int(s.max()):>10}")
    # tính bão hòa views (median % giá trị 72h đạt ở 6h/24h) để xác nhận tính chất
    F = views[:, -1].astype(float)
    m = F > 0
    r = views[m] / F[m, None]
    r6 = np.median(r[:, GRID.index(6)]) * 100
    r24 = np.median(r[:, GRID.index(24)]) * 100
    print(f"\n== bão hòa views (median) ==  @6h={r6:.1f}%  @24h={r24:.1f}%")


if __name__ == "__main__":
    main()
