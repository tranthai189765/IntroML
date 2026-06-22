"""
Demo: Reddit PA Pipeline với simulated multi-snapshot data.

Chạy:
    python run_reddit_data.py

So sánh 2 chế độ:
    run_static()   – 1 snapshot/post, không có temporal features
    run_temporal() – 11 snapshots/post (simulated), temporal features
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from src.reddit_pa.training.pipeline import RedditPAPipeline
from src.reddit_pa.training.temporal_sampler import RedditTemporalSampler
from src.reddit_pa.training.target_builder import RedditTargetBuilder, compute_reddit_score
from src.utils.evaluation import EvaluationReporter


# ---------------------------------------------------------------------------
# Simulated Reddit dataset (50 posts)
# ---------------------------------------------------------------------------

def make_reddit_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    subreddits = ["worldnews", "AskReddit", "gaming", "science", "movies",
                  "funny", "todayilearned", "technology", "nba", "pics"]
    flairs = ["Discussion", "News", "Question", "OC", "Shitpost", None]

    rows = []
    for i in range(n):
        sub = subreddits[i % len(subreddits)]
        score_final = int(rng.integers(10, 50_000))
        comments_final = int(rng.integers(1, 3_000))
        upvote_ratio = float(np.clip(rng.normal(0.75, 0.12), 0.4, 0.99))

        rows.append({
            "post_id":       f"reddit_{i:05d}",
            "title":         f"Sample Reddit post {i}: discussing {sub} topic number {i}",
            "selftext":      (
                f"This is the body of post {i} in r/{sub}. " * rng.integers(1, 5)
                if rng.random() > 0.4 else ""
            ),
            "subreddit":     sub,
            "score":         score_final,
            "num_comments":  comments_final,
            "upvote_ratio":  upvote_ratio,
            "created_utc":   1_700_000_000 + i * 3600,
            "link_flair_text": flairs[i % len(flairs)],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Static run
# ---------------------------------------------------------------------------

def run_static() -> dict:
    print("\n" + "="*60)
    print("STATIC PIPELINE (1 snapshot/post, no temporal features)")
    print("="*60)

    df = make_reddit_df(n=50)

    pipeline = RedditPAPipeline(
        title_dim=32, body_dim=32, subreddit_dim=16,
        regression_C=1.0, classification_C=1.0,
    )
    pipeline.fit(df)

    reg_hist = pipeline.regression_history
    cls_hist = pipeline.classification_history

    if not reg_hist or not cls_hist:
        print("  Not enough samples for evaluation.")
        return {}

    n_correct = sum(1 for r in cls_hist if r.correct)
    acc = n_correct / len(cls_hist)
    print(f"  Samples        : {len(df)}")
    print(f"  Feature dim    : {pipeline.feature_builder.feature_dim}")
    print(f"  Accuracy       : {acc:.1%}")

    avg_mae = np.mean([
        abs(r.y_pred_log[k] - r.y_true_log[k])
        for r in reg_hist
        for k in ["score", "num_comments", "upvote_ratio"]
    ])
    print(f"  MAE (log)      : {avg_mae:.4f}")

    return {"mode": "static", "accuracy": acc, "mae_log": float(avg_mae)}


# ---------------------------------------------------------------------------
# Temporal run
# ---------------------------------------------------------------------------

def run_temporal() -> dict:
    print("\n" + "="*60)
    print("TEMPORAL PIPELINE (11 snapshots/post simulated)")
    print("="*60)

    df = make_reddit_df(n=50)

    df_multi = RedditTemporalSampler.simulate_snapshots(df, crawl_time_col="created_utc")
    print(f"  Simulated rows : {len(df_multi)}  ({len(df)} posts × 11 snapshots)")

    pipeline = RedditPAPipeline(
        title_dim=32, body_dim=32, subreddit_dim=16,
        regression_C=1.0, classification_C=1.0,
        add_engagement=True,
    )

    sampler = RedditTemporalSampler()
    df_t, df_t1 = sampler.create_pairs(df_multi)
    print(f"  Training pairs : {len(df_t)}")

    pipeline.feature_builder.add_engagement = True
    Z = pipeline.feature_builder.fit_transform(df_t)
    print(f"  Feature dim    : {Z.shape[1]}")

    pipeline.target_builder.fit(df_t1)
    Y_reg = pipeline.target_builder.regression_targets(df_t1)
    y_cls = pipeline.target_builder.classification_labels(df_t1)

    pipeline.regression_head.run_online(Z, Y_reg)
    pipeline.classification_head.run_online(Z, y_cls)
    pipeline._trained = True

    reg_hist = pipeline.regression_history
    cls_hist = pipeline.classification_history

    n_correct = sum(1 for r in cls_hist if r.correct)
    acc = n_correct / len(cls_hist)

    mae_score = np.mean([
        abs(r.y_pred_log["score"] - r.y_true_log["score"])
        for r in reg_hist
    ])

    sampler.summary(df_t)
    print(f"  Accuracy       : {acc:.1%}")
    print(f"  MAE score(log) : {mae_score:.4f}")

    def reddit_score_fn(d: dict) -> np.ndarray:
        return compute_reddit_score(
            d.get("score",        np.zeros(1)),
            d.get("num_comments", np.zeros(1)),
            d.get("upvote_ratio", np.zeros(1)),
        )

    reporter = EvaluationReporter(
        reg_history=reg_hist,
        cls_history=cls_hist,
        score_fn=reddit_score_fn,
        target_names=["score", "num_comments", "upvote_ratio"],
        n_classes=4,
        label_names={0: "Low", 1: "Medium", 2: "Popular", 3: "Viral"},
    )
    reporter.print_all()

    return {"mode": "temporal", "accuracy": acc, "mae_score_log": float(mae_score)}


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def compare() -> None:
    r_static   = run_static()
    r_temporal = run_temporal()

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"  {'Mode':<20} {'Accuracy':>10} {'MAE(log)':>10}")
    print(f"  {'-'*40}")
    if r_static:
        print(f"  {'Static':<20} {r_static['accuracy']:>10.1%} {r_static['mae_log']:>10.4f}")
    if r_temporal:
        print(f"  {'Temporal':<20} {r_temporal['accuracy']:>10.1%} {r_temporal['mae_score_log']:>10.4f}")
    print()


if __name__ == "__main__":
    compare()
