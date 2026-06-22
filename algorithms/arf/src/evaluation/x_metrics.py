"""
src/evaluation/x_metrics.py
Đánh giá & trực quan hóa cho pipeline X/Twitter (multi-class ARF).

Gồm:
  - MulticlassHistory   : theo dõi lịch sử dự đoán theo cửa sổ trượt
  - RegressionHistory   : giống OnlineHistory nhưng gọn hơn
  - plot_x_learning_curves        : accuracy + macro-F1 theo thời gian
  - plot_x_confusion_matrix       : confusion matrix cuối stream
  - plot_x_label_distribution     : phân bố label trong dataset
  - plot_x_regression_scatter     : predicted vs actual score_6h
  - plot_x_score_trajectory       : trung bình score theo từng lớp label
  - save_x_metric_tables          : lưu CSV + in terminal
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import confusion_matrix

EVAL_WINDOW  = 200
WARM_UP_SIZE = 200
LABEL_NAMES  = ["Low", "Medium", "Popular", "Viral"]
LABEL_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

from src.models.arf_multiclass import REGRESSION_TARGETS

_REG_COLORS = {
    "likes_next":    "purple",
    "views_next":    "steelblue",
    "comments_next": "darkorange",
    "reposts_next":  "seagreen",
}
_REG_LABELS = {
    "likes_next":    "Likes (next snapshot)",
    "views_next":    "Views (next snapshot)",
    "comments_next": "Comments (next snapshot)",
    "reposts_next":  "Reposts (next snapshot)",
}


# ─────────────────────────────────────────────────────────────────────────────
# History Trackers
# ─────────────────────────────────────────────────────────────────────────────
class MulticlassHistory:
    """Theo dõi lịch sử dự đoán multi-class theo cửa sổ trượt."""

    def __init__(self, window: int = EVAL_WINDOW, n_classes: int = 4):
        self.window    = window
        self.n_classes = n_classes
        self.y_true: list = []
        self.y_pred: list = []
        self.n_seen = 0

    def update(self, y_pred: int, y_true: int) -> None:
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        self.n_seen += 1

    def windowed_accuracy(self) -> list[float]:
        accs = []
        for i in range(len(self.y_true)):
            start = max(0, i - self.window + 1)
            yt = self.y_true[start:i + 1]
            yp = self.y_pred[start:i + 1]
            accs.append(sum(a == b for a, b in zip(yt, yp)) / len(yt))
        return accs

    def windowed_macro_f1(self) -> list[float]:
        f1s = []
        for i in range(len(self.y_true)):
            start = max(0, i - self.window + 1)
            yt = self.y_true[start:i + 1]
            yp = self.y_pred[start:i + 1]
            class_f1 = []
            for c in range(self.n_classes):
                tp = sum(a == c and b == c for a, b in zip(yt, yp))
                fp = sum(a != c and b == c for a, b in zip(yt, yp))
                fn = sum(a == c and b != c for a, b in zip(yt, yp))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                class_f1.append(f1)
            f1s.append(float(np.mean(class_f1)))
        return f1s


class RegressionHistory:
    """Theo dõi lịch sử dự đoán multi-output regression theo cửa sổ trượt."""

    def __init__(self, window: int = EVAL_WINDOW):
        self.window = window
        self.y_true: list[dict] = []
        self.y_pred: list[dict] = []
        self.n_seen = 0

    def update(self, y_pred: dict, y_true: dict) -> None:
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        self.n_seen += 1

    def windowed_mae(self) -> dict[str, list[float]]:
        """Trả về {target: [mae tại mỗi bước]} cho cả 4 targets."""
        result = {t: [] for t in REGRESSION_TARGETS}
        for i in range(len(self.y_true)):
            start = max(0, i - self.window + 1)
            yt = self.y_true[start:i + 1]
            yp = self.y_pred[start:i + 1]
            for t in REGRESSION_TARGETS:
                result[t].append(float(np.mean([abs(a[t] - b[t]) for a, b in zip(yt, yp)])))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_x_learning_curves(
    clf_hist: MulticlassHistory,
    reg_hist: RegressionHistory,
    save_dir: str,
) -> None:
    """Đường cong học: Accuracy + Macro-F1 (classifier) và MAE (regressor)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Online Learning Curves – X/Twitter Dataset (Prequential, Window=200)",
                 fontsize=13)

    # ── Classification ────────────────────────────────────────────────────────
    ax = axes[0]
    accs = clf_hist.windowed_accuracy()
    f1s  = clf_hist.windowed_macro_f1()
    x    = range(len(accs))
    ax.plot(x, accs, label="Accuracy",    color="steelblue",  alpha=0.85, linewidth=1.2)
    ax.plot(x, f1s,  label="Macro-F1",    color="darkorange", alpha=0.85, linewidth=1.2)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6, label=f"warm-up end ({WARM_UP_SIZE})")
    ax.set_title("Virality Classifier (4-class ARF)")
    ax.set_xlabel("# Instances Evaluated (post warm-up)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Regression ────────────────────────────────────────────────────────────
    ax = axes[1]
    mae_per_target = reg_hist.windowed_mae()
    for t, maes in mae_per_target.items():
        ax.plot(range(len(maes)), maes,
                label=f"MAE {_REG_LABELS[t]}", color=_REG_COLORS[t],
                alpha=0.75, linewidth=1.1)
    ax.set_title("Engagement Regressor (ARF) — Predicting 4 metrics @6h")
    ax.set_xlabel("# Instances Evaluated (post warm-up)")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    _save(fig, os.path.join(save_dir, "x01_learning_curves.png"))


def plot_x_confusion_matrix(
    y_true: list, y_pred: list, save_dir: str
) -> None:
    """Confusion matrix đầy đủ cuối stream."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix – Virality Classifier (4-class ARF)", fontsize=13)

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Count", "Normalized (row %)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=ax, linewidths=0.5, cbar=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    _save(fig, os.path.join(save_dir, "x02_confusion_matrix.png"))


def plot_x_label_distribution(df: pd.DataFrame, save_dir: str) -> None:
    """Phân bố label và score_6h theo từng lớp."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Overview – X/Twitter (2,947 posts)", fontsize=13)

    # ── Label counts ─────────────────────────────────────────────────────────
    ax = axes[0]
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(
        [LABEL_NAMES[i] for i in counts.index],
        counts.values,
        color=LABEL_COLORS,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("Label Distribution (score_6h quantile)")
    ax.set_xlabel("Virality Label")
    ax.set_ylabel("# Posts")
    ax.grid(axis="y", alpha=0.3)

    # ── Score_6h distribution per label ──────────────────────────────────────
    ax = axes[1]
    for lbl, color in enumerate(LABEL_COLORS):
        subset = df[df["label"] == lbl]["score_6h"]
        ax.hist(subset, bins=30, alpha=0.6, color=color,
                label=LABEL_NAMES[lbl], density=True)
    ax.set_title("Score@6h Distribution per Label")
    ax.set_xlabel("score_6h")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    _save(fig, os.path.join(save_dir, "x03_label_distribution.png"))


def plot_x_regression_scatter(
    y_true: list[dict], y_pred: list[dict], save_dir: str
) -> None:
    """Predicted vs Actual — scatter 2×2 cho 4 engagement metrics @6h."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("ARF Regressor: Predicted vs Actual — Engagement @6h", fontsize=13)

    for ax, t in zip(axes.flat, REGRESSION_TARGETS):
        yt = np.array([d[t] for d in y_true])
        yp = np.array([d[t] for d in y_pred])

        mae  = np.mean(np.abs(yt - yp))
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ax.scatter(yt, yp, alpha=0.2, s=12, color=_REG_COLORS[t], edgecolors="none")
        lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
        pad = (hi - lo) * 0.05 + 1
        lim = [lo - pad, hi + pad]
        ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"Actual {_REG_LABELS[t]}")
        ax.set_ylabel(f"Predicted {_REG_LABELS[t]}")
        ax.set_title(_REG_LABELS[t])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.text(0.05, 0.90,
                f"MAE={mae:.1f}   RMSE={rmse:.1f}   R²={r2:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    _save(fig, os.path.join(save_dir, "x04_regression_scatter.png"))


def plot_x_feature_importance_proxy(df: pd.DataFrame, save_dir: str) -> None:
    """
    Proxy feature importance: Spearman correlation của mỗi feature với score_6h.
    (ARF online không có built-in feature importance, dùng correlation thay thế.)
    """
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        return

    corr = (
        df[feat_cols + ["score_6h"]]
        .corr(method="spearman")["score_6h"]
        .drop("score_6h")
        .abs()
        .sort_values(ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#C44E52" if v > 0.3 else "#4C72B0" for v in corr.values]
    bars = ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_title("Top-20 Feature Importance (|Spearman ρ| with score_6h)", fontsize=12)
    ax.set_xlabel("|Spearman ρ|")
    ax.axvline(0.3, color="gray", linestyle="--", alpha=0.7, label="ρ=0.3 threshold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    _save(fig, os.path.join(save_dir, "x05_feature_importance.png"))


def plot_x_topic_ranking(
    early_df: pd.DataFrame,
    final_df: pd.DataFrame,
    group_col: str,
    save_dir: str,
    top_n: int = 15,
) -> None:
    """
    So sánh Topic Ranking tại 0.5h (early) vs 6h (final).

    Công thức: Topic_Score_K = Σ (Base_Score_i × W_tier_i)
    W: Viral=5, Popular=3, Medium=1, Low=0
    """
    e = early_df.head(top_n).copy()
    f = final_df.head(top_n).copy()

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, top_n * 0.45)))
    fig.suptitle(
        f"Topic Ranking — Topic_Score = Σ(Base_Score × W_tier)\n"
        f"W: Viral=5.0 | Popular=3.0 | Medium=1.0 | Low=0.0",
        fontsize=12,
    )

    # ── Early @0.5h ──────────────────────────────────────────────────────────
    ax = axes[0]
    topics_e = e[group_col].tolist()[::-1]
    scores_e = e["topic_score_05h"].tolist()[::-1]
    bars_e = ax.barh(topics_e, scores_e, color="#4C72B0", alpha=0.85)
    ax.set_title("Early @0.5h (used as FEATURE — no leak)", fontsize=11)
    ax.set_xlabel("Topic_Score (Σ score × W_tier)")
    for bar, val in zip(bars_e, scores_e):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # ── Final @6h ────────────────────────────────────────────────────────────
    ax = axes[1]
    topics_f = f[group_col].tolist()[::-1]
    scores_f = f["topic_score_final"].tolist()[::-1]
    bars_f = ax.barh(topics_f, scores_f, color="#C44E52", alpha=0.85)
    ax.set_title("Final @6h (GROUND TRUTH — report only)", fontsize=11)
    ax.set_xlabel("Topic_Score (Σ score × W_tier)")
    for bar, val in zip(bars_f, scores_f):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    _save(fig, os.path.join(save_dir, "x07_topic_ranking.png"))


def plot_x_author_follower_vs_label(df: pd.DataFrame, save_dir: str) -> None:
    """Box plot: author_log_followers theo từng lớp label."""
    fig, ax = plt.subplots(figsize=(9, 6))
    data = [df[df["label"] == lbl]["author_log_followers"].values for lbl in range(4)]
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], LABEL_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(LABEL_NAMES)
    ax.set_title("Author Follower Count (log) per Virality Label\n"
                 "(Mỗi lớp gấp ~10–16× lớp dưới — tín hiệu virality mạnh nhất)", fontsize=11)
    ax.set_xlabel("Virality Label")
    ax.set_ylabel("ln(1 + followers)")
    ax.grid(axis="y", alpha=0.3)

    # Median annotation
    medians = [float(np.median(d)) for d in data]
    for i, med in enumerate(medians):
        ax.text(i + 1, med + 0.15, f"med={med:.1f}", ha="center", fontsize=8, color="black")

    _save(fig, os.path.join(save_dir, "x06_author_follower_vs_label.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Summary Tables
# ─────────────────────────────────────────────────────────────────────────────
def save_x_metric_tables(
    clf_metrics: dict,
    reg_metrics: dict,
    y_true_clf:  list,
    y_pred_clf:  list,
    tables_dir:  str,
) -> None:
    # ── Classification table ─────────────────────────────────────────────────
    clf_rows = [[
        "Virality Classifier (4-class ARF)",
        f"{clf_metrics['accuracy']:.4f}",
        f"{clf_metrics['macro_f1']:.4f}",
        f"{clf_metrics['weighted_f1']:.4f}",
        f"{clf_metrics['kappa']:.4f}",
        clf_metrics['n_seen'],
    ]]
    clf_df = pd.DataFrame(clf_rows,
        columns=["Task", "Accuracy", "Macro-F1", "Weighted-F1", "Kappa", "N Seen"])
    clf_df.to_csv(os.path.join(tables_dir, "x_classification_metrics.csv"), index=False)

    print("\n-- Classification Metrics (Final – after warm-up) -----------------")
    print(tabulate(clf_df, headers="keys", tablefmt="grid"))

    # Per-class report
    cm = confusion_matrix(y_true_clf, y_pred_clf, labels=[0, 1, 2, 3])
    per_class = []
    for lbl in range(4):
        tp  = cm[lbl, lbl]
        fp  = cm[:, lbl].sum() - tp
        fn  = cm[lbl, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        supp = cm[lbl, :].sum()
        per_class.append([LABEL_NAMES[lbl], f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", int(supp)])

    pc_df = pd.DataFrame(per_class, columns=["Class", "Precision", "Recall", "F1", "Support"])
    pc_df.to_csv(os.path.join(tables_dir, "x_per_class_metrics.csv"), index=False)
    print("\n-- Per-Class Metrics -----------------------------------------------")
    print(tabulate(pc_df, headers="keys", tablefmt="grid"))

    # ── Regression table ──────────────────────────────────────────────────────
    n_seen = reg_metrics["n_seen"]
    reg_rows = [
        [
            _REG_LABELS[t],
            f"{reg_metrics[t]['mae']:.2f}",
            f"{reg_metrics[t]['rmse']:.2f}",
            f"{reg_metrics[t]['r2']:.4f}",
            n_seen,
        ]
        for t in REGRESSION_TARGETS
    ]
    reg_df = pd.DataFrame(reg_rows, columns=["Target", "MAE", "RMSE", "R²", "N Seen"])
    reg_df.to_csv(os.path.join(tables_dir, "x_regression_metrics.csv"), index=False)

    print("\n-- Regression Metrics (Final — 4 targets @6h) -----------------------")
    print(tabulate(reg_df, headers="keys", tablefmt="grid"))
    print("-" * 62)
