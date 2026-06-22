"""
run_arf_72h.py
Chạy Adaptive Random Forest (river) trên ĐÚNG dữ liệu 72h + embeddings có sẵn của nhóm
(dataset_72h.csv + embeddings_72h), GRID 13 mốc, để so sánh với PA và online SGD.

- KHÔNG embed lại ảnh (bỏ ResNet18). Dùng image_emb_per_post / text_emb / author_emb
  đã tính sẵn (SigLIP/BGE), rút gọn PCA cho ARF chạy được.
- Prequential (test-then-train) online stream theo SNAPSHOT_PAIRS (12 cặp, 0.5h..72h).
- Classification 4 lớp + Regression 4 target (raw-scale MAE/RMSE + theo từng label).

ENV:
  N_POSTS   : giới hạn số post (mặc định: tất cả)  -> dùng để ước lượng thời gian
  WARMUP    : số instance khởi động trước khi đánh giá (mặc định 1000)
  PCA_IMG/PCA_TXT/PCA_AUTH : số chiều PCA (mặc định 24/24/16)
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             mean_absolute_error, mean_squared_error)
from tqdm import tqdm

from src.features.x_extractor import SNAPSHOT_PAIRS, build_stream_instance
from src.models.arf_multiclass import ViralityClassifier, MultiOutputRegressor, REGRESSION_TARGETS

DATA   = os.environ.get("CSV_PATH", os.path.join("data", "dataset_72h.csv"))
EMBDIR = os.environ.get("EMB_DIR", os.path.join("data", "embeddings_72h"))
N_POSTS = int(os.environ.get("N_POSTS", "0"))         # 0 = tất cả
WARMUP  = int(os.environ.get("WARMUP", "1000"))
PCA_IMG = int(os.environ.get("PCA_IMG", "24"))
PCA_TXT = int(os.environ.get("PCA_TXT", "24"))
PCA_AUTH= int(os.environ.get("PCA_AUTH", "16"))
GROUP_COL = "lang"
TARGET_RAW = {"likes_next": "likes", "views_next": "views",
              "comments_next": "comments", "reposts_next": "reposts"}
LABEL_NAMES = {0: "Low", 1: "Medium", 2: "Popular", 3: "Viral"}


def add_pca(df, arr, k, prefix):
    k = min(k, arr.shape[1])
    z = PCA(n_components=k, random_state=42).fit_transform(arr).astype(np.float32)
    for j in range(k):
        df[f"feat_emb_{prefix}{j}"] = z[:, j]


def main():
    t0 = time.time()
    df_full = pd.read_csv(DATA)
    N = len(df_full)
    if N_POSTS and N_POSTS < N:
        rng = np.random.default_rng(42)
        labels_full = df_full["label"].astype(int).to_numpy()
        sel = []
        for lab in np.unique(labels_full):
            idx_lab = np.where(labels_full == lab)[0]
            k = max(1, round(len(idx_lab) * N_POSTS / N))   # giữ tỉ lệ nhãn
            sel.append(rng.choice(idx_lab, size=min(k, len(idx_lab)), replace=False))
        sel = np.sort(np.concatenate(sel))
    else:
        sel = np.arange(N)
    df = df_full.iloc[sel].reset_index(drop=True)
    n = len(df)
    print(f"[data] {n}/{N} posts (stratified) | GRID pairs = "
          f"{[(o, nx) for o, _, nx, _ in SNAPSHOT_PAIRS]}")

    # ── embeddings có sẵn (căn theo thứ tự id, index ĐÚNG theo mẫu đã chọn) ─────
    txt = np.load(os.path.join(EMBDIR, "text_emb.npy"))[sel]
    aut = np.load(os.path.join(EMBDIR, "author_emb.npy"))[sel]
    img = np.load(os.path.join(EMBDIR, "image_emb_per_post.npy"))[sel]
    print(f"[emb ] text{txt.shape} author{aut.shape} image{img.shape} -> PCA "
          f"{PCA_TXT}/{PCA_AUTH}/{PCA_IMG}")
    add_pca(df, img, PCA_IMG,  "img")
    add_pca(df, txt, PCA_TXT,  "txt")
    add_pca(df, aut, PCA_AUTH, "auth")

    # ── static feature có sẵn trong dataset_72h ────────────────────────────────
    df["feat_has_image"]    = df["has_image"].astype(int)
    df["feat_has_video"]    = df["has_video"].astype(int)
    df["feat_intake_age_h"] = pd.to_numeric(df["intake_age_h"], errors="coerce").fillna(0.0)
    df["feat_log_text_len"] = np.log1p(df["text"].fillna("").astype(str).str.len())

    final_label = df["label"].astype(int).to_numpy()   # nhãn cuối (72h) cho per-label
    post_ids = df["id"].astype(str).to_numpy()

    # ── stream snapshot-centric: (obs_h, idx) ──────────────────────────────────
    stream = []
    for idx, row in df.iterrows():
        for obs_suffix, obs_h, next_suffix, next_h in SNAPSHOT_PAIRS:
            stream.append((idx, obs_h, row, obs_suffix, next_suffix, next_h))
    stream.sort(key=lambda s: (s[1], s[0]))
    n_inst = len(stream)
    print(f"[stream] {n_inst:,} instances ({n} x {len(SNAPSHOT_PAIRS)}) | warmup={WARMUP}")

    clf = ViralityClassifier()
    reg = MultiOutputRegressor()

    yt_c, yp_c, lab_eval = [], [], []
    yt_r = {t: [] for t in REGRESSION_TARGETS}
    yp_r = {t: [] for t in REGRESSION_TARGETS}
    arf_oof = []   # (post_id, pred nhãn-cuối, true nhãn-cuối) ở bước 60h->72h

    for si, (idx, obs_h, row, obs_suffix, next_suffix, next_h) in enumerate(tqdm(
            stream, total=n_inst, ncols=70, desc="  ARF",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")):
        x = build_stream_instance(row, obs_suffix, obs_h, next_h, topic_features=None)
        y_label = int(row[f"label_{next_suffix}"])
        y_score = {f"{tk}_next": float(np.log1p(float(row[f"{rv}_{next_suffix}"])))
                   for tk, rv in [("likes", "likes"), ("views", "views"),
                                  ("comments", "comments"), ("reposts", "reposts")]}

        pred_label = clf.predict_one(x)
        pred_score = reg.predict_one(x)

        if obs_suffix == "60h":   # cặp 60h->72h: dự đoán nhãn-CUỐI per-post (test-then-train)
            arf_oof.append((post_ids[idx], int(pred_label), int(final_label[idx])))

        if si >= WARMUP:
            yt_c.append(y_label); yp_c.append(pred_label); lab_eval.append(final_label[idx])
            for t in REGRESSION_TARGETS:
                yt_r[t].append(y_score[t]); yp_r[t].append(pred_score[t])

        clf.learn_one(x, y_label)
        reg.learn_one(x, y_score)

    # ── METRICS ────────────────────────────────────────────────────────────────
    yt_c = np.array(yt_c); yp_c = np.array(yp_c); lab_eval = np.array(lab_eval)
    acc = accuracy_score(yt_c, yp_c)
    f1m = f1_score(yt_c, yp_c, average="macro", zero_division=0)
    prm = precision_score(yt_c, yp_c, average="macro", zero_division=0)
    rcm = recall_score(yt_c, yp_c, average="macro", zero_division=0)

    print("\n" + "=" * 64)
    print(f"ARF (river) — PREQUENTIAL trên data 72h, {len(yt_c):,} mẫu eval")
    print("=" * 64)
    print("Classification (label 0..3):")
    print(f"  accuracy        : {acc:.4f}")
    print(f"  f1_macro        : {f1m:.4f}")
    print(f"  precision_macro : {prm:.4f}")
    print(f"  recall_macro    : {rcm:.4f}")

    # raw-scale regression (expm1 để cùng đơn vị với PA/SGD)
    print("Regression (raw-scale error of NEXT snapshot):")
    raw_pred = {}; raw_true = {}
    for t in REGRESSION_TARGETS:
        pr = np.expm1(np.array(yp_r[t])); tr = np.expm1(np.array(yt_r[t]))
        pr = np.clip(pr, 0, None)
        raw_pred[t] = pr; raw_true[t] = tr
        mae = mean_absolute_error(tr, pr); rmse = float(np.sqrt(mean_squared_error(tr, pr)))
        print(f"  {TARGET_RAW[t]:9}: MAE {mae:>10.3f}   RMSE {rmse:>11.2f}")

    print("\nRegression MAE/RMSE theo TỪNG LABEL (raw-scale):")
    for lab in (0, 1, 2, 3):
        m = lab_eval == lab
        if m.sum() == 0:
            continue
        print(f"  --- label {lab} ({LABEL_NAMES[lab]}), n_eval={int(m.sum())} ---")
        for t in REGRESSION_TARGETS:
            mae = mean_absolute_error(raw_true[t][m], raw_pred[t][m])
            rmse = float(np.sqrt(mean_squared_error(raw_true[t][m], raw_pred[t][m])))
            print(f"    {TARGET_RAW[t]:9}: MAE {mae:>10.2f}   RMSE {rmse:>11.2f}")

    import pandas as _pd
    oof = _pd.DataFrame(arf_oof, columns=["post_id", "pred_label", "true_label"])
    oof.to_csv("oof_pred_arf_72h.csv", index=False)
    print(f"[oof] saved {len(oof)} per-post final-label predictions -> oof_pred_arf_72h.csv")

    print(f"\n[done] {time.time()-t0:.1f}s  ({n_inst/(time.time()-t0):.0f} inst/s)")


if __name__ == "__main__":
    main()
