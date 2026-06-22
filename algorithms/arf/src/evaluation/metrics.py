"""
src/evaluation/metrics.py
Đánh giá hiệu suất Online Learning theo phương pháp Prequential
(Interleaved Test-Then-Train).

Prequential Evaluation:
  Với mỗi mẫu thứ i:
    1. Dự đoán với model hiện tại → ghi nhận error
    2. Cập nhật model với true label
  → Metric được tính cộng dồn (cumulative) hoặc trên cửa sổ trượt (windowed).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # không cần GUI
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import EVAL_WINDOW, WARM_UP_SIZE, TOPICS, TOP_K_TOPICS


# ─────────────────────────────────────────────────────────────────────────────
# Online History Tracker
# ─────────────────────────────────────────────────────────────────────────────
class OnlineHistory:
    """Lưu lịch sử dự đoán theo từng bước để vẽ đồ thị."""

    def __init__(self, window: int = EVAL_WINDOW):
        self.window  = window
        self.y_true  : list = []
        self.y_pred  : list = []
        self.n_seen  : int  = 0

    def update(self, y_pred, y_true):
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        self.n_seen += 1

    def windowed_accuracy(self) -> list[float]:
        accs = []
        for i in range(len(self.y_true)):
            start = max(0, i - self.window + 1)
            yt = self.y_true[start : i + 1]
            yp = self.y_pred[start : i + 1]
            acc = sum(a == b for a, b in zip(yt, yp)) / len(yt)
            accs.append(acc)
        return accs

    def windowed_f1(self) -> list[float]:
        """Windowed macro-F1 for binary classification."""
        f1s = []
        for i in range(len(self.y_true)):
            start = max(0, i - self.window + 1)
            yt = self.y_true[start : i + 1]
            yp = self.y_pred[start : i + 1]
            tp = sum(a == 1 and b == 1 for a, b in zip(yt, yp))
            fp = sum(a == 0 and b == 1 for a, b in zip(yt, yp))
            fn = sum(a == 1 and b == 0 for a, b in zip(yt, yp))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1s.append(f1)
        return f1s

    def windowed_mae(self) -> list[float]:
        """Windowed MAE for regression."""
        maes = []
        for i in range(len(self.y_true)):
            start = max(0, i - self.window + 1)
            yt = self.y_true[start : i + 1]
            yp = self.y_pred[start : i + 1]
            mae = np.mean([abs(a - b) for a, b in zip(yt, yp)])
            maes.append(mae)
        return maes


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_learning_curves(
    topic_hist: OnlineHistory,
    post_hist:  OnlineHistory,
    reg_hist:   OnlineHistory,
    save_dir:   str,
) -> None:
    """Vẽ đường cong học (accuracy/F1/MAE) theo thời gian stream."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Online Learning Curves (Prequential – Windowed)", fontsize=14)

    # ── Topic Classifier ────────────────────────────────────────────────────
    ax = axes[0]
    accs = topic_hist.windowed_accuracy()
    f1s  = topic_hist.windowed_f1()
    x    = range(len(accs))
    ax.plot(x, accs, label="Accuracy", color="steelblue",  alpha=0.8)
    ax.plot(x, f1s,  label="F1",       color="darkorange", alpha=0.8)
    ax.axvline(WARM_UP_SIZE, color="gray", linestyle="--", label="warm-up end")
    ax.set_title("Table 1: Topic Popularity")
    ax.set_xlabel("# Samples Seen")
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # ── Post Classifier ─────────────────────────────────────────────────────
    ax = axes[1]
    accs = post_hist.windowed_accuracy()
    f1s  = post_hist.windowed_f1()
    x    = range(len(accs))
    ax.plot(x, accs, label="Accuracy", color="seagreen",  alpha=0.8)
    ax.plot(x, f1s,  label="F1",       color="firebrick", alpha=0.8)
    ax.axvline(WARM_UP_SIZE, color="gray", linestyle="--", label="warm-up end")
    ax.set_title("Table 2: Post Popularity in Topic")
    ax.set_xlabel("# Samples Seen")
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # ── Regressor ───────────────────────────────────────────────────────────
    ax = axes[2]
    maes = reg_hist.windowed_mae()
    x    = range(len(maes))
    ax.plot(x, maes, label="MAE (engagement_target)", color="purple", alpha=0.8)
    ax.axvline(WARM_UP_SIZE, color="gray", linestyle="--", label="warm-up end")
    ax.set_title("Regression: Engagement Target at 24h")
    ax.set_xlabel("# Samples Seen")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.grid(alpha=0.3)

    _save(fig, os.path.join(save_dir, "01_learning_curves.png"))


def plot_topic_ranking_table1(df: pd.DataFrame, save_dir: str) -> None:
    """Vẽ bảng xếp hạng topic (Table 1) dựa trên hot_score tại 1h và 24h."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Table 1: Topic Popularity Ranking", fontsize=14)

    for ax, snapshot, color in zip(
        axes, ["1h", "24h"], ["steelblue", "darkorange"]
    ):
        topic_agg = (
            df.groupby("topic")[f"hot_score_{snapshot}"]
            .sum()
            .sort_values(ascending=True)
        )
        bars = ax.barh(topic_agg.index, topic_agg.values, color=color, alpha=0.8)
        ax.set_title(f"Total Hot Score @ {snapshot} snapshot")
        ax.set_xlabel("Total Hot Score")

        # Đánh dấu top-K
        top_topics = topic_agg.nlargest(TOP_K_TOPICS).index
        for bar, lbl in zip(bars, topic_agg.index):
            if lbl in top_topics:
                bar.set_edgecolor("red")
                bar.set_linewidth(2.5)

        ax.grid(axis="x", alpha=0.3)

    _save(fig, os.path.join(save_dir, "02_table1_topic_ranking.png"))


def plot_post_ranking_table2(df: pd.DataFrame, save_dir: str) -> None:
    """
    Vẽ Table 2: Top posts per topic.
    Stacked bar: % popular posts vs non-popular posts trong từng topic.
    """
    topic_pop = (
        df.groupby("topic")["is_popular_post"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["firebrick" if t in topic_pop.head(TOP_K_TOPICS).index
              else "steelblue" for t in topic_pop.index]
    bars = ax.bar(topic_pop.index, topic_pop.values, color=colors, alpha=0.8)

    ax.set_title("Table 2: % Posts Classified as 'Popular' per Topic", fontsize=13)
    ax.set_ylabel("% Popular Posts")
    ax.set_xlabel("Topic")
    ax.axhline(10, color="gray", linestyle="--", label="threshold 10%")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, topic_pop.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    _save(fig, os.path.join(save_dir, "03_table2_post_ranking.png"))


def plot_ranking_algorithm_comparison(df: pd.DataFrame, save_dir: str) -> None:
    """So sánh 5 thuật toán ranking bằng Spearman correlation."""
    score_cols = [
        "hot_score_1h", "reddit_score_1h", "weighted_score_1h",
        "velocity_score_1h", "wilson_score_1h",
    ]
    labels = ["Hacker News", "Reddit Hot", "Weighted Sum", "Velocity", "Wilson LB"]

    # Correlation matrix (Spearman rank correlation)
    corr = df[score_cols].corr(method="spearman")
    corr.index   = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
                vmin=0.5, vmax=1.0, ax=ax, square=True,
                linewidths=0.5, cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Ranking Algorithm Correlation (Spearman ρ)", fontsize=13)

    _save(fig, os.path.join(save_dir, "04_ranking_algorithm_comparison.png"))


def plot_regression_scatter(y_true: list, y_pred: list, save_dir: str) -> None:
    """Scatter plot: predicted vs actual engagement_target."""
    yt = np.array(y_true)
    yp = np.array(y_pred)

    # Clip extreme values for visibility
    p99 = np.percentile(yt, 99)
    mask = yt <= p99
    yt, yp = yt[mask], yp[mask]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(yt, yp, alpha=0.15, s=10, color="steelblue")
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Engagement Target (24h)")
    ax.set_ylabel("Predicted Engagement Target (24h)")
    ax.set_title("Regression: Predicted vs Actual Engagement")
    ax.legend()
    ax.grid(alpha=0.3)

    mae  = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    ax.text(0.05, 0.92, f"MAE={mae:.1f}  RMSE={rmse:.1f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    _save(fig, os.path.join(save_dir, "05_regression_scatter.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Summary Tables
# ─────────────────────────────────────────────────────────────────────────────
def save_metric_tables(
    topic_metrics: dict,
    post_metrics:  dict,
    reg_metrics:   dict,
    topic_rank:    pd.DataFrame,
    tables_dir:    str,
) -> None:
    """Lưu bảng kết quả dạng CSV và in ra terminal."""

    # ── Bảng 1: Classification metrics ────────────────────────────────────────
    clf_rows = [
        ["Topic Classifier (Table 1)", topic_metrics.get("accuracy", 0),
         topic_metrics.get("f1", 0), topic_metrics.get("kappa", 0)],
        ["Post Classifier  (Table 2)", post_metrics.get("accuracy", 0),
         post_metrics.get("f1", 0), post_metrics.get("kappa", 0)],
    ]
    clf_df = pd.DataFrame(clf_rows, columns=["Task", "Accuracy", "F1", "Cohen Kappa"])
    clf_df.to_csv(os.path.join(tables_dir, "classification_metrics.csv"), index=False)

    print("\n-- Classification Metrics (Final) ----------------------------------")
    print(tabulate(clf_df, headers="keys", tablefmt="grid", floatfmt=".4f"))

    # Regression metrics
    reg_rows = [["Engagement Regressor",
                 reg_metrics.get("mae", 0),
                 reg_metrics.get("rmse", 0),
                 reg_metrics.get("r2", 0)]]
    reg_df = pd.DataFrame(reg_rows, columns=["Task", "MAE", "RMSE", "R2"])
    reg_df.to_csv(os.path.join(tables_dir, "regression_metrics.csv"), index=False)

    print("\n-- Regression Metrics (Final) --------------------------------------")
    print(tabulate(reg_df, headers="keys", tablefmt="grid", floatfmt=".4f"))

    # Topic Ranking
    topic_rank.to_csv(os.path.join(tables_dir, "topic_ranking.csv"), index=False)

    print("\n-- Table 1: Topic Ranking (based on 24h hot_score) ----------------")
    print(tabulate(topic_rank.head(10), headers="keys",
                   tablefmt="grid", floatfmt=".2f"))
    print("-" * 62)
