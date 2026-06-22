"""
main_x.py
Pipeline Online Learning trên bộ dữ liệu X/Twitter thực (data/X_data/dataset_aligned.csv).

Luồng xử lý:
  1. Load & làm sạch dữ liệu (2,947 bài X/Twitter, 7 mốc thời gian)
  2. Trích xuất image embedding (ResNet18 pretrained → PCA 16-dim, cached)
  3. Ranking per Topic:
       Topic_Score_K = Σ (score_0_5h_i × W_tier_i)
       W: Viral=5.0 | Popular=3.0 | Medium=1.0 | Low=0.0
       (group by 'lang'; early @0.5h → feature; final @6h → report only)
  4. Trích xuất features (tabular + topic context + image embedding)
  5. Chạy prequential evaluation (test-then-train, ARF):
       ├── Task A: 4-class Virality Classification
       └── Task B: Score Regression – dự đoán score_6h
  6. Lưu kết quả & vẽ 8 đồ thị báo cáo

Ranking per Posts:
  Score = ln(0.01·V + L + 5·C + 10·R + 1)   [đã có sẵn trong dataset]
  Labels: 3=Viral(top5%) | 2=Popular(5–20%) | 1=Medium(20–50%) | 0=Low(bottom50%)
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from tqdm import tqdm
from config import RANDOM_SEED

# ── Paths ─────────────────────────────────────────────────────────────────────
X_DATA_DIR    = os.path.join(ROOT, "data", "X_data")
X_MEDIA_DIR   = os.path.join(X_DATA_DIR, "media")
X_EMB_CACHE   = os.path.join(X_DATA_DIR, "image_embeddings.npz")

X_RESULTS_DIR = os.path.join(ROOT, "results", "X_data")
X_FIGURES_DIR = os.path.join(X_RESULTS_DIR, "figures")
X_TABLES_DIR  = os.path.join(X_RESULTS_DIR, "tables")

WARM_UP_SIZE = 200
GROUP_COL    = "lang"    # proxy "topic" trong X dataset


def setup_dirs() -> None:
    for d in [X_RESULTS_DIR, X_FIGURES_DIR, X_TABLES_DIR]:
        os.makedirs(d, exist_ok=True)


def print_banner() -> None:
    print("=" * 65)
    print("  X/TWITTER VIRALITY PREDICTION")
    print("  Online Learning with Adaptive Random Forest (ARF)")
    print("  Real dataset: data/X_data/dataset_aligned.csv")
    print("  4-class Classification + Score Regression")
    print("  Ranking: Topic_Score = Σ(score × W_tier)")
    print("=" * 65)


def main() -> None:
    setup_dirs()
    print_banner()
    t0 = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 1: Load dữ liệu
    # ─────────────────────────────────────────────────────────────────────────
    from src.data.x_loader import load_x_dataset, print_dataset_summary

    print("\n[1/6] Loading X/Twitter dataset ...")
    df = load_x_dataset(seed=RANDOM_SEED)
    print_dataset_summary(df)
    df.to_csv(os.path.join(X_RESULTS_DIR, "x_raw_loaded.csv"), index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 2: Image Embedding (ResNet18 → PCA 32-dim)
    # ─────────────────────────────────────────────────────────────────────────
    from src.features.image_embedder import build_image_embeddings

    print("\n[2/6] Extracting image embeddings (ResNet18 pretrained → PCA 16-dim) ...")
    print(f"      Media dir : {X_MEDIA_DIR}")
    print(f"      Cache     : {X_EMB_CACHE}")

    img_emb_df = build_image_embeddings(
        df,
        media_dir    = X_MEDIA_DIR,
        cache_path   = X_EMB_CACHE,
        n_components = 16,
    )
    print(f"      → {len(img_emb_df.columns) - 1} image PCA features ready")

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 3: Topic Ranking
    # ─────────────────────────────────────────────────────────────────────────
    from src.ranking.x_scorer import (
        compute_early_topic_scores,
        compute_final_topic_scores,
        print_topic_ranking,
        TIER_WEIGHTS,
    )

    print(f"\n[3/6] Computing Topic Rankings (group_col='{GROUP_COL}') ...")
    print(f"      Formula  : Topic_Score_K = Σ (score_i × W_tier_i)")
    print(f"      W_tier   : {TIER_WEIGHTS}")

    topic_early_df = compute_early_topic_scores(df, group_col=GROUP_COL)
    topic_final_df = compute_final_topic_scores(df, group_col=GROUP_COL)

    print_topic_ranking(topic_early_df, topic_final_df, group_col=GROUP_COL, top_n=15)

    # Lưu bảng ranking
    topic_early_df.to_csv(os.path.join(X_TABLES_DIR, "x_topic_ranking_early.csv"), index=False)
    topic_final_df.to_csv(os.path.join(X_TABLES_DIR, "x_topic_ranking_final.csv"), index=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 4: Feature Extraction
    # ─────────────────────────────────────────────────────────────────────────
    from src.features.x_extractor import (
        extract_x_features, X_FEATURE_COLS,
        SNAPSHOT_PAIRS, build_stream_instance,
    )

    print("\n[4/6] Extracting features (tabular + topic context + image PCA) ...")
    # Topic context được cấp per-snapshot qua TopicStatsTracker trong stream loop.
    df = extract_x_features(
        df,
        img_emb_df = img_emb_df,
        group_col  = GROUP_COL,
    )
    n_topic_feat = len([c for c in X_FEATURE_COLS if "topic" in c])
    n_img_feat   = len([c for c in X_FEATURE_COLS if "img" in c])
    n_tab_feat   = len(X_FEATURE_COLS) - n_topic_feat - n_img_feat
    print(f"      → {len(X_FEATURE_COLS)} features total:")
    print(f"        {n_tab_feat} tabular  |  {n_topic_feat} topic context  |  {n_img_feat} image PCA")

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 5: Prequential Evaluation (Online Learning Stream)
    # ─────────────────────────────────────────────────────────────────────────
    from src.models.arf_multiclass import ViralityClassifier, MultiOutputRegressor
    from src.evaluation.x_metrics  import MulticlassHistory, RegressionHistory
    from src.ranking.x_scorer      import TopicStatsTracker

    # ── Xây dựng next-step stream ─────────────────────────────────────────────
    # Mỗi post × 6 bước: obs_suffix → next_suffix (ví dụ 0.5h→1h, 1h→1.5h, ...)
    # Snapshot-centric: toàn bộ bài @0.5h trước, rồi @1h, ...
    # Sort key = (obs_h, orig_idx) → obs_h làm khóa chính.
    # Lý do: mô phỏng thực tế (hệ thống quan sát tất cả bài tại cùng 1 mốc),
    # giúp TopicStatsTracker tích lũy context đầy đủ trong mỗi batch, và ADWIN
    # phát hiện đúng drift phân phối khi chuyển từ batch này sang batch khác.
    stream: list[tuple[int, float, object, str, float, str, float]] = []
    for orig_idx, row in df.iterrows():
        for obs_suffix, obs_h, next_suffix, next_h in SNAPSHOT_PAIRS:
            stream.append((orig_idx, obs_h, row, obs_suffix, obs_h, next_suffix, next_h))
    stream.sort(key=lambda s: (s[1], s[0]))

    n_posts     = len(df)
    n_steps     = len(SNAPSHOT_PAIRS)
    n_instances = len(stream)

    print(f"\n[5/6] Running prequential next-step stream ...")
    print(f"      Posts     : {n_posts:,}")
    print(f"      Steps     : {[(o, nx) for o, _, nx, _ in SNAPSHOT_PAIRS]}")
    print(f"      Total     : {n_instances:,} instances  ({n_posts} × {n_steps})")
    print(f"      Warm-up   : first {WARM_UP_SIZE} instances (train only)\n")

    clf = ViralityClassifier()
    reg = MultiOutputRegressor()

    clf_hist = MulticlassHistory(window=200)
    reg_hist = RegressionHistory(window=200)

    y_true_clf: list = []
    y_pred_clf: list = []
    y_true_reg: list[dict] = []
    y_pred_reg: list[dict] = []

    topic_tracker = TopicStatsTracker()

    for stream_idx, (_, obs_h, row, obs_suffix, _, next_suffix, next_h) in enumerate(tqdm(
        stream, total=n_instances, desc="  Streaming",
        ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    )):
        topic_feats = topic_tracker.get_features(obs_suffix, row[GROUP_COL])
        x = build_stream_instance(row, obs_suffix, obs_h, next_h, topic_feats)

        y_label = int(row[f"label_{next_suffix}"])
        y_score = {
            "likes_next":    np.log1p(float(row[f"likes_{next_suffix}"])),
            "views_next":    np.log1p(float(row[f"views_{next_suffix}"])),
            "comments_next": np.log1p(float(row[f"comments_{next_suffix}"])),
            "reposts_next":  np.log1p(float(row[f"reposts_{next_suffix}"])),
        }

        # ── Prequential: Predict BEFORE Train ─────────────────────────────────
        pred_label = clf.predict_one(x)
        pred_score = reg.predict_one(x)

        # ── Evaluate (after warm-up) ───────────────────────────────────────────
        if stream_idx >= WARM_UP_SIZE:
            clf.update_metrics(pred_label, y_label)
            reg.update_metrics(pred_score, y_score)

            clf_hist.update(pred_label, y_label)
            reg_hist.update(pred_score, y_score)

            y_true_clf.append(y_label)
            y_pred_clf.append(pred_label)
            y_true_reg.append(y_score)
            y_pred_reg.append(pred_score)

        # ── Train ──────────────────────────────────────────────────────────────
        clf.learn_one(x, y_label)
        reg.learn_one(x, y_score)

        # Cập nhật topic stats cho ĐÚNG snapshot vừa quan sát (đồng tuổi, leak-safe)
        topic_tracker.update(obs_suffix, row[GROUP_COL], x["feat_score"], int(row[f"label_{obs_suffix}"]))

    # ── Save model ─────────────────────────────────────────────────────────────
    import pickle
    MODEL_DIR = os.path.join(X_RESULTS_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "clf.pkl"),           "wb") as f: pickle.dump(clf,           f)
    with open(os.path.join(MODEL_DIR, "reg.pkl"),           "wb") as f: pickle.dump(reg,           f)
    with open(os.path.join(MODEL_DIR, "topic_tracker.pkl"), "wb") as f: pickle.dump(topic_tracker, f)
    print(f"  Models saved → {MODEL_DIR}/")

    # ─────────────────────────────────────────────────────────────────────────
    # Bước 6: Kết quả & Visualization
    # ─────────────────────────────────────────────────────────────────────────
    from src.evaluation.x_metrics import (
        plot_x_learning_curves,
        plot_x_confusion_matrix,
        plot_x_label_distribution,
        plot_x_regression_scatter,
        plot_x_feature_importance_proxy,
        plot_x_topic_ranking,
        plot_x_author_follower_vs_label,
        save_x_metric_tables,
    )

    print(f"\n[6/6] Saving results & generating report figures ...")

    save_x_metric_tables(
        clf_metrics = clf.get_metrics(),
        reg_metrics = reg.get_metrics(),
        y_true_clf  = y_true_clf,
        y_pred_clf  = y_pred_clf,
        tables_dir  = X_TABLES_DIR,
    )

    print("\n  Generating figures ...")
    plot_x_learning_curves(clf_hist, reg_hist, X_FIGURES_DIR)
    plot_x_confusion_matrix(y_true_clf, y_pred_clf, X_FIGURES_DIR)
    plot_x_label_distribution(df, X_FIGURES_DIR)
    plot_x_regression_scatter(y_true_reg, y_pred_reg, X_FIGURES_DIR)
    plot_x_feature_importance_proxy(df, X_FIGURES_DIR)
    plot_x_topic_ranking(topic_early_df, topic_final_df, GROUP_COL, X_FIGURES_DIR)
    plot_x_author_follower_vs_label(df, X_FIGURES_DIR)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Figures  → {X_FIGURES_DIR}/")
    print(f"  Tables   → {X_TABLES_DIR}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
