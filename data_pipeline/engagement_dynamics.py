# -*- coding: utf-8 -*-
"""
Phân tích động học tương tác theo thời gian để biện minh cho lịch snapshot.
Sinh:
  - fig_saturation.(png|pdf) : đường bão hòa (% giá trị mốc 72h) + vận tốc tăng trưởng
  - fig_trajectories.(png|pdf): quỹ đạo tích lũy chuẩn hóa của một mẫu post
  - engagement_dynamics.csv   : bảng số liệu % đạt được & vận tốc theo từng mốc
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PLATFORM = os.environ.get("PLATFORM", "twitter").lower()
CSV = os.environ.get("CSV_PATH", "data_v1/dataset_72h.csv")
OUT = os.environ.get("OUT_DIR", "data_v1")
TAG = os.environ.get("TAG", "" if PLATFORM == "twitter" else "_" + PLATFORM)
GRID = [0.5, 1, 1.5, 2, 3, 4, 6, 10, 16, 24, 48, 60, 72]
T = np.array(GRID, dtype=float)


def gtag(g):
    return str(g).replace(".", "_")


if PLATFORM == "reddit":
    PNAME = "Reddit"
    METRICS = [
        ("views",     "Lượt xem", "#2563eb"),
        ("upvotes",   "Upvote",   "#16a34a"),
        ("downvotes", "Downvote", "#dc2626"),
    ]
else:
    PNAME = "X/Twitter"
    METRICS = [
        ("views",    "Lượt xem",  "#2563eb"),
        ("likes",    "Lượt thích", "#dc2626"),
        ("reposts",  "Repost",    "#16a34a"),
        ("comments", "Bình luận", "#9333ea"),
    ]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220,
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
})


def load_series(df, metric):
    cols = [f"{metric}_{gtag(g)}h" for g in GRID]
    return df[cols].to_numpy(dtype=float)  # (n, T)


def reached_ratio(arr):
    """% giá trị mốc cuối (72h) đạt được tại mỗi mốc; lọc post có final>0."""
    F = arr[:, -1]
    m = F > 0
    r = arr[m] / F[m, None]            # (n_valid, T)
    r = np.clip(r, 0, 1.5)
    return r


def main():
    df = pd.read_csv(CSV)
    n = len(df)
    print(f"[data] {n} posts, {len(GRID)} snapshots: {GRID}")

    # ---- thống kê % đạt được + vận tốc ----
    stats = {"snapshot_h": GRID}
    med_curve = {}
    for metric, _, _ in METRICS:
        arr = load_series(df, metric)
        r = reached_ratio(arr)
        med = np.median(r, axis=0) * 100
        q25 = np.percentile(r, 25, axis=0) * 100
        q75 = np.percentile(r, 75, axis=0) * 100
        med_curve[metric] = (med, q25, q75)
        stats[f"{metric}_pct_median"] = np.round(med, 2)
        # vận tốc: % giá trị cuối tăng thêm mỗi giờ (giữa 2 mốc liên tiếp)
        dv = np.diff(med)
        dt = np.diff(T)
        vel = np.concatenate([[np.nan], dv / dt])
        stats[f"{metric}_vel_pct_per_h"] = np.round(vel, 3)

    sdf = pd.DataFrame(stats)
    sdf.to_csv(os.path.join(OUT, f"engagement_dynamics{TAG}.csv"), index=False)

    # in tóm tắt cho report
    def at(metric, h):
        i = GRID.index(h)
        return med_curve[metric][0][i]
    print("\n== % giá trị mốc 72h đã đạt được (median) ==")
    print(f"{'metric':9} {'@1h':>7} {'@6h':>7} {'@24h':>7} {'@48h':>7}")
    for metric, name, _ in METRICS:
        print(f"{metric:9} {at(metric,1):6.1f}% {at(metric,6):6.1f}% "
              f"{at(metric,24):6.1f}% {at(metric,48):6.1f}%")
    print("\n== Tỉ trọng tăng trưởng theo cửa sổ (views, median) ==")
    vm = med_curve["views"][0] / 100
    g06 = vm[GRID.index(6)] - 0.0
    g624 = vm[GRID.index(24)] - vm[GRID.index(6)]
    g2472 = vm[-1] - vm[GRID.index(24)]
    print(f"  0–6h  : {g06*100:5.1f}% tổng tăng trưởng")
    print(f"  6–24h : {g624*100:5.1f}%")
    print(f"  24–72h: {g2472*100:5.1f}%")

    # ===================== FIG 1: saturation + velocity =====================
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    # vùng nền: tăng mạnh / chuyển tiếp / bão hòa
    spans = [(0.4, 6, "#16a34a", "Tăng mạnh (0–6h)"),
             (6, 24, "#f59e0b", "Chuyển tiếp (6–24h)"),
             (24, 80, "#94a3b8", "Bão hòa (>24h)")]
    for ax in (axA, axB):
        for x0, x1, c, _ in spans:
            ax.axvspan(x0, x1, color=c, alpha=0.07, lw=0)
        ax.set_xscale("log")
        ax.set_xlim(0.45, 80)
        ax.set_xticks(GRID)
        ax.set_xticklabels([("%g" % g) for g in GRID], fontsize=8.5)
        ax.set_xlabel("Tuổi bài đăng (giờ, thang log)")

    # --- Panel A: % đạt được ---
    for metric, name, c in METRICS:
        med, q25, q75 = med_curve[metric]
        axA.plot(T, med, "-o", color=c, lw=2.2, ms=5, label=name, zorder=3)
        if metric == "views":
            axA.fill_between(T, q25, q75, color=c, alpha=0.13, lw=0, zorder=1)
    axA.axhline(100, color="#334155", ls=":", lw=1, alpha=0.6)
    axA.set_ylabel("% giá trị mốc 72h đã đạt được (median)")
    axA.set_ylim(0, 108)
    axA.set_title("Bão hòa tương tác: phần lớn đạt được rất sớm")
    # chú thích mốc
    v6 = at("views", 6); v24 = at("views", 24)
    bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    axA.annotate(f"6h: ≈{v6:.0f}% lượt xem", xy=(6, v6), xytext=(0.95, 30),
                 fontsize=9.5, color="#15803d", fontweight="bold", bbox=bbox,
                 arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.3))
    axA.annotate(f"24h: ≈{v24:.0f}%", xy=(24, v24), xytext=(33, 55),
                 fontsize=9.5, color="#b45309", fontweight="bold", bbox=bbox,
                 arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.3))
    axA.legend(loc="lower right", frameon=False, fontsize=10)

    # --- Panel B: vận tốc (%/h, log) ---
    mid = np.sqrt(T[:-1] * T[1:])         # midpoint trên thang log
    for metric, name, c in METRICS:
        med = med_curve[metric][0]
        vel = np.diff(med) / np.diff(T)
        vel = np.where(vel > 1e-2, vel, np.nan)   # ngừng vẽ khi đã bão hòa (≈0)
        axB.plot(mid, vel, "-o", color=c, lw=2.0, ms=4.5, label=name, zorder=3)
    axB.set_yscale("log")
    axB.set_ylim(8e-3, None)
    axB.set_ylabel("Vận tốc tăng trưởng (% giá trị cuối / giờ, log)")
    axB.set_title("Vận tốc sụp đổ nhanh sau 6h")
    axB.legend(loc="upper right", frameon=False, fontsize=10)

    # legend vùng nền chung
    handles = [Patch(facecolor=c, alpha=0.18, label=lab) for _, _, c, lab in spans]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Động học tương tác của bài đăng %s" % PNAME,
                 fontsize=14.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_saturation{TAG}.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # ===================== FIG 2: spaghetti trajectories =====================
    arr = load_series(df, "views")
    r = reached_ratio(arr) * 100
    rng = np.random.default_rng(42)
    idx = rng.choice(r.shape[0], size=min(300, r.shape[0]), replace=False)
    fig2, ax = plt.subplots(figsize=(8, 5.2))
    for x0, x1, c, _ in spans:
        ax.axvspan(x0, x1, color=c, alpha=0.07, lw=0)
    for i in idx:
        ax.plot(T, r[i], color="#2563eb", alpha=0.05, lw=0.8)
    med, q25, q75 = (np.median(r, 0), np.percentile(r, 25, 0), np.percentile(r, 75, 0))
    ax.fill_between(T, q25, q75, color="#1d4ed8", alpha=0.18, lw=0, label="Khoảng IQR (25–75%)")
    ax.plot(T, med, "-o", color="#0b2a6b", lw=2.6, ms=5, label="Trung vị")
    ax.set_xscale("log"); ax.set_xlim(0.45, 80)
    ax.set_xticks(GRID); ax.set_xticklabels([("%g" % g) for g in GRID], fontsize=8.5)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Tuổi bài đăng (giờ, thang log)")
    ax.set_ylabel("% lượt xem mốc 72h đã đạt được")
    ax.set_title("Quỹ đạo tích lũy lượt xem %s (chuẩn hóa)" % PNAME)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(os.path.join(OUT, f"fig_trajectories{TAG}.{ext}"), bbox_inches="tight")
    plt.close(fig2)

    print(f"\n[done] saved: {OUT}/fig_saturation{TAG}.(png|pdf), "
          f"{OUT}/fig_trajectories{TAG}.(png|pdf), {OUT}/engagement_dynamics{TAG}.csv")


if __name__ == "__main__":
    main()
