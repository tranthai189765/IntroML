"""
test_x.py
Đánh giá ARF trên tập held-out (20% cuối) — cùng seed & pipeline với main_x.py.

Luồng:
  1. Load dữ liệu (cùng seed với main_x.py)
  2. Split 80/20: train_df / test_df
  3. Trích xuất features (author/content/image)
  4. Train prequential trên train split (chỉ train, không đánh giá)
  5. Predict trên test split (không train thêm)
  6. In metrics: Accuracy, Macro-F1, Kappa, per-class + MAE, RMSE, R² per target
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

sys.stdout.reconfigure(encoding="utf-8")

# CRITICAL: Import image_embedder (which imports torch) BEFORE sklearn to avoid
# OpenMP run-time DLL initialization conflict (WinError 1114) on Windows.
from src.features.image_embedder import build_image_embeddings
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from config import RANDOM_SEED
from src.data.x_loader import load_x_dataset
from src.features.x_extractor import (
    extract_x_features, SNAPSHOT_PAIRS, build_stream_instance,
)
from src.models.arf_multiclass import (
    ViralityClassifier, MultiOutputRegressor, REGRESSION_TARGETS,
)
from src.ranking.x_scorer import TopicStatsTracker

# ── Config ─────────────────────────────────────────────────────────────────────
X_DATA_DIR  = os.path.join(ROOT, "data", "X_data")
X_MEDIA_DIR = os.path.join(X_DATA_DIR, "media")
X_EMB_CACHE = os.path.join(X_DATA_DIR, "image_embeddings.npz")
RESULTS_DIR = os.path.join(ROOT, "results", "X_data", "test")
GROUP_COL   = "lang"
TRAIN_RATIO = 0.8

LABEL_NAMES = ["Low", "Medium", "Popular", "Viral"]
REG_LABELS  = {
    "likes_next":    "Likes   (next snapshot)",
    "views_next":    "Views   (next snapshot)",
    "comments_next": "Comments(next snapshot)",
    "reposts_next":  "Reposts (next snapshot)",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_stream(df: pd.DataFrame) -> list:
    """Snapshot-centric multi-snapshot stream — toàn bộ bài @obs_h trước, rồi bước tiếp theo."""
    items = []
    for orig_idx, row in df.iterrows():
        for obs_suffix, obs_h, next_suffix, next_h in SNAPSHOT_PAIRS:
            items.append((orig_idx, obs_h, row, obs_suffix, obs_h, next_suffix, next_h))
    items.sort(key=lambda s: (s[1], s[0]))
    return items


def make_y_score(row: pd.Series, next_suffix: str) -> dict:
    return {
        "likes_next":    np.log1p(float(row[f"likes_{next_suffix}"])),
        "views_next":    np.log1p(float(row[f"views_{next_suffix}"])),
        "comments_next": np.log1p(float(row[f"comments_{next_suffix}"])),
        "reposts_next":  np.log1p(float(row[f"reposts_{next_suffix}"])),
    }


def clf_metrics(y_true: list, y_pred: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (summary_df, per_class_df)."""
    yt = np.array(y_true)
    yp = np.array(y_pred)
    n  = len(yt)

    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])

    per_class_rows = []
    f1s = []
    for lbl in range(4):
        tp   = int(cm[lbl, lbl])
        fp   = int(cm[:, lbl].sum()) - tp
        fn   = int(cm[lbl, :].sum()) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        supp = int(cm[lbl, :].sum())
        f1s.append(f1)
        per_class_rows.append(
            [LABEL_NAMES[lbl], f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", supp]
        )

    acc        = float((yt == yp).mean())
    macro_f1   = float(np.mean(f1s))
    w_f1       = float(sum(f1s[i] * cm[i, :].sum() for i in range(4)) / n)
    p_e        = sum((cm[i, :].sum() * cm[:, i].sum()) for i in range(4)) / (n ** 2)
    kappa      = (acc - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    summary = pd.DataFrame([[
        "Virality Classifier (ARF)",
        f"{acc:.4f}", f"{macro_f1:.4f}", f"{w_f1:.4f}", f"{kappa:.4f}", n,
    ]], columns=["Task", "Accuracy", "Macro-F1", "Weighted-F1", "Kappa", "N"])

    per_class = pd.DataFrame(
        per_class_rows,
        columns=["Class", "Precision", "Recall", "F1", "Support"],
    )
    return summary, per_class


def reg_metrics(y_true: list[dict], y_pred: list[dict]) -> pd.DataFrame:
    rows = []
    for t in REGRESSION_TARGETS:
        yt = np.array([d[t] for d in y_true])
        yp = np.array([d[t] for d in y_pred])
        mae  = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        ss_r = float(np.sum((yt - yp) ** 2))
        ss_t = float(np.sum((yt - yt.mean()) ** 2))
        r2   = 1 - ss_r / ss_t if ss_t > 0 else 0.0
        rows.append([REG_LABELS[t], f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}", len(yt)])
    return pd.DataFrame(rows, columns=["Target", "MAE", "RMSE", "R²", "N"])


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 65)
    print("  X/TWITTER — HELD-OUT TEST EVALUATION")
    print(f"  Split : {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% test")
    print(f"  Stream: snapshot-centric, {len(SNAPSHOT_PAIRS)} batches × all posts (next-step)")
    print("=" * 65)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset ...")
    df = load_x_dataset(seed=RANDOM_SEED)

    n_train  = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df  = df.iloc[n_train:].reset_index(drop=True)
    print(f"      Total : {len(df):,} posts")
    print(f"      Train : {len(train_df):,}  |  Test: {len(test_df):,}")

    # ── 2. Image embeddings ────────────────────────────────────────────────────
    # PCA được fit CHỈ trên train_df (fit_df=train_df) để tránh leakage.
    # df đầy đủ được truyền để transform cả train lẫn test trong một lần,
    # nhưng PCA không thấy test images khi fit.
    print("\n[2/4] Extracting features ...")
    img_emb_df = build_image_embeddings(
        df, media_dir=X_MEDIA_DIR, cache_path=X_EMB_CACHE,
        n_components=16, fit_df=train_df,
    )
    train_df = extract_x_features(train_df, img_emb_df=img_emb_df, group_col=GROUP_COL)
    test_df  = extract_x_features(test_df,  img_emb_df=img_emb_df, group_col=GROUP_COL)
    print("      Static features (author / content / image) ready.")

    # ── 3. Load model hoặc retrain ─────────────────────────────────────────────
    import pickle
    MODEL_DIR  = os.path.join(ROOT, "results", "X_data", "test", "models")
    CLF_PATH   = os.path.join(MODEL_DIR, "clf_train_only.pkl")
    REG_PATH   = os.path.join(MODEL_DIR, "reg_train_only.pkl")
    TOPIC_PATH = os.path.join(MODEL_DIR, "topic_tracker_train_only.pkl")

    if os.path.exists(CLF_PATH) and os.path.exists(REG_PATH) and os.path.exists(TOPIC_PATH):
        print(f"\n[3/4] Loading train-only saved model from {MODEL_DIR}/ ...")
        with open(CLF_PATH,   "rb") as f: clf           = pickle.load(f)
        with open(REG_PATH,   "rb") as f: reg           = pickle.load(f)
        with open(TOPIC_PATH, "rb") as f: topic_tracker = pickle.load(f)
        print("      clf, reg, topic_tracker loaded (leak-free).")
    else:
        n_train_inst = len(train_df) * len(SNAPSHOT_PAIRS)
        print(f"\n[3/4] No train-only model found — retraining on {len(train_df):,} posts"
              f" × {len(SNAPSHOT_PAIRS)} steps = {n_train_inst:,} instances ...")

        clf           = ViralityClassifier()
        reg           = MultiOutputRegressor()
        topic_tracker = TopicStatsTracker()

        for _, obs_h, row, obs_suffix, _, next_suffix, next_h in tqdm(
            build_stream(train_df), total=n_train_inst,
            desc="  Training", ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        ):
            topic_feats = topic_tracker.get_features(obs_suffix, row[GROUP_COL])
            x           = build_stream_instance(row, obs_suffix, obs_h, next_h, topic_feats)
            y_label     = int(row[f"label_{next_suffix}"])
            y_score     = make_y_score(row, next_suffix)

            clf.learn_one(x, y_label)
            reg.learn_one(x, y_score)

            topic_tracker.update(
                obs_suffix, row[GROUP_COL], x["feat_score"], int(row[f"label_{obs_suffix}"])
            )

        # Lưu model train-only để chạy lần sau không cần training lại
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(CLF_PATH,   "wb") as f: pickle.dump(clf,           f)
        with open(REG_PATH,   "wb") as f: pickle.dump(reg,           f)
        with open(TOPIC_PATH, "wb") as f: pickle.dump(topic_tracker, f)
        print(f"      Train-only models saved to → {MODEL_DIR}/")

    # ── 4. Test phase ──────────────────────────────────────────────────────────
    n_test_inst = len(test_df) * len(SNAPSHOT_PAIRS)
    print(f"\n[4/4] Evaluating on {len(test_df):,} posts × {len(SNAPSHOT_PAIRS)} steps"
          f" = {n_test_inst:,} instances ...")

    y_true_clf: list       = []
    y_pred_clf: list       = []
    y_true_reg: list[dict] = []
    y_pred_reg: list[dict] = []

    for _, obs_h, row, obs_suffix, _, next_suffix, next_h in tqdm(
        build_stream(test_df), total=n_test_inst,
        desc="  Testing ", ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    ):
        topic_feats = topic_tracker.get_features(obs_suffix, row[GROUP_COL])
        x           = build_stream_instance(row, obs_suffix, obs_h, next_h, topic_feats)

        pred_label = clf.predict_one(x)
        pred_score = reg.predict_one(x)

        y_true_clf.append(int(row[f"label_{next_suffix}"]))
        y_pred_clf.append(pred_label if pred_label is not None else 0)
        y_true_reg.append(make_y_score(row, next_suffix))
        y_pred_reg.append(pred_score)

    # ── 5. Kết quả ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  TEST RESULTS  —  {n_test_inst:,} instances  ({elapsed:.1f}s)")
    print(f"{'='*65}")

    clf_sum, clf_pc = clf_metrics(y_true_clf, y_pred_clf)
    reg_df          = reg_metrics(y_true_reg, y_pred_reg)

    print("\n-- Classification (held-out test) ----------------------------------")
    print(tabulate(clf_sum, headers="keys", tablefmt="grid", showindex=False))
    print("\n-- Per-Class Metrics -----------------------------------------------")
    print(tabulate(clf_pc,  headers="keys", tablefmt="grid", showindex=False))
    print("\n-- Regression (held-out test) — log1p scale ------------------------")
    print(tabulate(reg_df,  headers="keys", tablefmt="grid", showindex=False))
    print("-" * 65)

    # Save CSV
    clf_sum.to_csv(os.path.join(RESULTS_DIR, "test_clf_summary.csv"),   index=False)
    clf_pc.to_csv( os.path.join(RESULTS_DIR, "test_clf_per_class.csv"), index=False)
    reg_df.to_csv( os.path.join(RESULTS_DIR, "test_reg_metrics.csv"),   index=False)
    print(f"\n  Saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
