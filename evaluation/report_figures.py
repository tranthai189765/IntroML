# -*- coding: utf-8 -*-
"""Sinh figure cho chapter4 (kết quả học trực tuyến). Lưu PDF+PNG vào data_v1/."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "font.size": 11})
OUT = "data_v1"
C = {"PA": "#2563eb", "SGD": "#f59e0b", "ARF": "#16a34a"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ── 1) Phân loại 4 lớp ─────────────────────────────────────────────────────────
def fig_cls():
    metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    data = {"PA": [0.885, 0.771, 0.835, 0.769],
            "SGD": [0.883, 0.652, 0.681, 0.683],
            "ARF": [0.948, 0.926, 0.935, 0.919]}
    x = np.arange(len(metrics)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (m, v) in enumerate(data.items()):
        b = ax.bar(x + (i - 1) * w, v, w, label=m, color=C[m])
        ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(["Accuracy", "F1-macro", "Precision-macro", "Recall-macro"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Giá trị")
    ax.set_title("Phân loại độ viral 4 lớp")
    ax.legend(frameon=False, ncol=3); ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_cls_compare")


# ── 2) Hồi quy: views MAE theo label (log) ─────────────────────────────────────
def fig_reg():
    labs = ["Low", "Medium", "Popular", "Viral"]
    data = {"PA": [27.3, 242.8, 514.1, 165.6],
            "SGD": [14.6, 149.5, 171.7, 505.2],
            "ARF": [30.6, 585.2, 12249, 387186]}
    x = np.arange(len(labs)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (m, v) in enumerate(data.items()):
        b = ax.bar(x + (i - 1) * w, v, w, label=m, color=C[m])
        ax.bar_label(b, fmt="%.0f", fontsize=8, padding=2)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("MAE lượt xem (thang log)")
    ax.set_title("Sai số hồi quy lượt xem theo từng nhãn")
    ax.legend(frameon=False, ncol=3); ax.grid(axis="y", alpha=0.25, which="both")
    save(fig, "fig_reg_perlabel")


# ── 3) Topic ranking (ground-truth thật, tô theo tier) ─────────────────────────
def fig_topic():
    CONTRIB = {0: 0, 1: 1, 2: 3, 3: 5}
    t = pd.read_csv("data_v1/topic_features.csv")[["id", "topic"]].dropna()
    l = pd.read_csv("data_v1/dataset_72h.csv")[["id", "label"]]
    d = t.merge(l, on="id"); d["c"] = d["label"].astype(int).map(CONTRIB)
    g = d.groupby("topic").agg(score=("c", "sum")).reset_index().sort_values("score")
    n = len(g); ranks = np.arange(n, 0, -1)  # 1 = cao nhất
    cv = max(1, round(.05 * n)); cp = max(cv, round(.20 * n)); cm = max(cp, round(.50 * n))
    tier = np.where(ranks <= cv, 3, np.where(ranks <= cp, 2, np.where(ranks <= cm, 1, 0)))
    col = {3: "#ef4444", 2: "#f59e0b", 1: "#60a5fa", 0: "#9ca3af"}
    colors = [col[x] for x in tier]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(g["topic"], g["score"], color=colors)
    ax.set_xlabel("Điểm viral chủ đề  (Σ: label3→5, 2→3, 1→1, 0→0)")
    ax.set_title("Xếp hạng chủ đề theo độ viral (thực tế)")
    leg = [Patch(color=col[k], label=f"{nm}") for k, nm in
           [(3, "Viral (top 5%)"), (2, "Popular (5–20%)"), (1, "Medium (20–50%)"), (0, "Low (50% cuối)")]]
    ax.legend(handles=leg, frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "fig_topic_ranking")


# ── 4) Online vs Offline (bar accuracy, tô theo regime) ────────────────────────
def fig_onoff():
    rows = [("online-ARF", 0.905, "ONLINE"), ("online-PA", 0.884, "ONLINE"),
            ("online-SGD", 0.867, "ONLINE"), ("offline-retrain", 0.896, "OFFLINE"),
            ("offline-frozen", 0.828, "OFFLINE")]
    names = [r[0] for r in rows]; vals = [r[1] for r in rows]
    cols = ["#16a34a" if r[2] == "ONLINE" else "#dc2626" for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    b = ax.bar(names, vals, color=cols, width=0.62)
    ax.bar_label(b, fmt="%.3f", fontsize=9, padding=2)
    ax.set_ylim(0.78, 0.93); ax.set_ylabel("Accuracy (per-post)")
    ax.set_title("Online vs Offline learning")
    ax.legend(handles=[Patch(color="#16a34a", label="Online (vừa dự đoán vừa học)"),
                       Patch(color="#dc2626", label="Offline (baseline)")],
              frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25); plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    save(fig, "fig_online_offline")


for f in (fig_cls, fig_reg, fig_topic, fig_onoff):
    f()
print("[done] data_v1/fig_cls_compare, fig_reg_perlabel, fig_topic_ranking, fig_online_offline (pdf+png)")
