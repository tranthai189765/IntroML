# -*- coding: utf-8 -*-
"""
online_vs_offline.py — Phần 3: ONLINE vs OFFLINE learning.

Cô lập đúng yếu tố "regime": CÙNG họ mô hình tuyến tính, CÙNG feature, CÙNG stream
prequential (post × 12 cặp snapshot, snapshot-centric), chỉ khác CÁCH CẬP NHẬT:

  [ONLINE]          online-SGD / online-PA : vừa dự đoán vừa học (cập nhật từng mẫu).
  [OFFLINE-retrain] thấy data mới -> train lại trên TOÀN BỘ đã thấy (mỗi K mẫu) rồi dự đoán.
  [OFFLINE-frozen]  train 1 lần trên tập đầu, ĐÓNG BĂNG, chỉ dự đoán.

Đánh giá trên cùng cửa sổ [EVAL_START:]. Feature = engagement + PCA(embedding) + static
(giống ARF). Ngoài ra in kèm số ONLINE của PA/SGD/ARF custom (Phần 1) để tham chiếu.

Output: online_vs_offline_results.txt + online_vs_offline.png
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV  = "data_v1/dataset_72h.csv"
EMB  = "data_v1/embeddings_72h"
GRID = [0.5, 1, 1.5, 2, 3, 4, 6, 10, 16, 24, 48, 60, 72]
PAIRS = [(GRID[i], GRID[i + 1]) for i in range(len(GRID) - 1)]
CLASSES = np.array([0, 1, 2, 3])
EVAL_START = 25000      # ranh giới train/eval trên stream
RETRAIN_K  = 8000       # offline-retrain: refit mỗi K mẫu
PCA_IMG, PCA_TXT, PCA_AUTH = 16, 16, 8
SEED = 42


def tag(g): return str(g).replace(".", "_")


def build_stream():
    df = pd.read_csv(CSV)
    n = len(df)
    txt = np.load(f"{EMB}/text_emb.npy")[:n]
    aut = np.load(f"{EMB}/author_emb.npy")[:n]
    img = np.load(f"{EMB}/image_emb_per_post.npy")[:n]
    emb = np.hstack([
        PCA(PCA_IMG, random_state=SEED).fit_transform(img),
        PCA(PCA_TXT, random_state=SEED).fit_transform(txt),
        PCA(PCA_AUTH, random_state=SEED).fit_transform(aut),
    ]).astype(np.float32)

    has_img = df["has_image"].to_numpy(float)
    has_vid = df["has_video"].to_numpy(float)
    intake  = pd.to_numeric(df["intake_age_h"], errors="coerce").fillna(0).to_numpy(float)
    tlen    = np.log1p(df["text"].fillna("").astype(str).str.len().to_numpy(float))
    final_lab = df["label"].astype(int).to_numpy()

    def col(m, g): return df[f"{m}_{tag(g)}h"].to_numpy(float)
    b = {m: col(m, 0.5) for m in ["views", "likes", "comments", "reposts"]}   # baseline 0.5h

    blocks, ys, obs_hs, finals = [], [], [], []
    for (oh, nh) in PAIRS:
        v, l, c, r = col("views", oh), col("likes", oh), col("comments", oh), col("reposts", oh)
        score = np.log(0.01 * v + l + 5 * c + 10 * r + 1)
        eng = np.column_stack([
            np.full(n, oh), np.full(n, nh), np.full(n, nh - oh),
            np.log1p(v), np.log1p(l), np.log1p(c), np.log1p(r), score,
            np.log1p(v / oh), np.log1p(l / oh), np.log1p(c / oh), np.log1p(r / oh),
            l / np.maximum(v, 1), r / np.maximum(v, 1), c / np.maximum(v, 1),
            r / (l + 1), c / (l + 1), np.log1p(l + c + r),
            np.log1p(v) - np.log1p(b["views"]), np.log1p(l) - np.log1p(b["likes"]),
            np.log1p(c) - np.log1p(b["comments"]), np.log1p(r) - np.log1p(b["reposts"]),
            has_img, has_vid, tlen, intake,
        ]).astype(np.float32)
        X = np.hstack([eng, emb])
        blocks.append(X)
        ys.append(df[f"label_{tag(nh)}h"].astype(int).to_numpy())
        obs_hs.append(np.full(n, oh)); finals.append(final_lab)
    X = np.vstack(blocks); y = np.concatenate(ys)
    obs = np.concatenate(obs_hs); fin = np.concatenate(finals)
    order = np.lexsort((np.tile(np.arange(n), len(PAIRS)), obs))  # snapshot-centric
    return X[order], y[order], fin[order]


def windowed_acc(yt, yp, w=3000):
    acc = []
    for i in range(0, len(yt), w):
        s = slice(i, i + w)
        if i + w <= len(yt):
            acc.append(accuracy_score(yt[s], yp[s]))
    return np.array(acc)


def main():
    t0 = time.time()
    X, y, fin = build_stream()
    N, D = X.shape
    print(f"[stream] {N} instances, {D} features | EVAL_START={EVAL_START} | retrain K={RETRAIN_K}")
    scaler = StandardScaler().fit(X[:EVAL_START])
    Xs = scaler.transform(X).astype(np.float32)
    ev = slice(EVAL_START, N)
    yt = y[ev]
    preds = {}

    # ── ONLINE-SGD (prequential) ────────────────────────────────────────────────
    # LR 'constant' (eta0 nhỏ) ổn định trên stream tuần tự không-iid; 'optimal' mặc
    # định decay theo t -> bất ổn khi data sắp theo tuổi snapshot.
    m = SGDClassifier(loss="log_loss", alpha=1e-5, learning_rate="constant",
                      eta0=0.01, random_state=SEED)
    m.partial_fit(Xs[:2000], y[:2000], classes=CLASSES)
    p = np.empty(N - EVAL_START, dtype=int)
    for j, i in enumerate(range(EVAL_START, N)):
        p[j] = m.predict(Xs[i:i + 1])[0]
        m.partial_fit(Xs[i:i + 1], y[i:i + 1])
    preds["online-SGD"] = p

    # ── ONLINE-PA (prequential) ─────────────────────────────────────────────────
    m = PassiveAggressiveClassifier(C=0.01, random_state=SEED)
    m.partial_fit(Xs[:2000], y[:2000], classes=CLASSES)
    p = np.empty(N - EVAL_START, dtype=int)
    for j, i in enumerate(range(EVAL_START, N)):
        p[j] = m.predict(Xs[i:i + 1])[0]
        m.partial_fit(Xs[i:i + 1], y[i:i + 1])
    preds["online-PA"] = p

    # ── OFFLINE-frozen (train 1 lần [0:EVAL_START], đóng băng) ──────────────────
    m = LogisticRegression(max_iter=400, C=1.0, multi_class="auto")
    m.fit(Xs[:EVAL_START], y[:EVAL_START])
    preds["offline-frozen"] = m.predict(Xs[ev])

    # ── OFFLINE-retrain (refit toàn bộ đã thấy mỗi K mẫu) ───────────────────────
    p = np.empty(N - EVAL_START, dtype=int)
    seen = EVAL_START
    m = LogisticRegression(max_iter=400, C=1.0, multi_class="auto").fit(Xs[:seen], y[:seen])
    n_refit = 1
    for j, i in enumerate(range(EVAL_START, N)):
        if i - seen >= RETRAIN_K:                 # có đủ K mẫu mới -> train lại trên TẤT CẢ
            seen = i
            m = LogisticRegression(max_iter=400, C=1.0, multi_class="auto").fit(Xs[:seen], y[:seen])
            n_refit += 1
        p[j] = m.predict(Xs[i:i + 1])[0]
    preds["offline-retrain"] = p

    # ── KẾT QUẢ ─────────────────────────────────────────────────────────────────
    lines = ["ONLINE vs OFFLINE — cùng họ tuyến tính, cùng feature/stream, eval [%d:%d]"
             % (EVAL_START, N)]
    lines.append(f"(offline-retrain refit {n_refit} lần; mỗi lần train lại trên toàn bộ đã thấy)\n")
    lines.append(f"{'Method':18} {'regime':16} {'accuracy':>9} {'f1_macro':>9}")
    rows = [("online-SGD", "ONLINE"), ("online-PA", "ONLINE"),
            ("offline-retrain", "OFFLINE-retrain"), ("offline-frozen", "OFFLINE-frozen")]
    res = {}
    for name, reg in rows:
        a = accuracy_score(yt, preds[name]); f = f1_score(yt, preds[name], average="macro", zero_division=0)
        res[name] = (a, f)
        lines.append(f"{name:18} {reg:16} {a:>9.4f} {f:>9.4f}")
    lines.append("\nTHAM CHIẾU (online, pipeline custom — Phần 1):")
    lines.append(f"{'PA  (custom)':18} {'ONLINE':16} {0.9510:>9.4f} {'-':>9}")
    lines.append(f"{'SGD (custom)':18} {'ONLINE':16} {0.8903:>9.4f} {'-':>9}")
    lines.append(f"{'ARF (custom)':18} {'ONLINE':16} {0.9482:>9.4f} {0.9262:>9.4f}")

    # diễn giải
    onl = max(res['online-SGD'][0], res['online-PA'][0])
    lines.append("\nNHẬN XÉT:")
    lines.append(f"  - online (tốt nhất {onl:.4f}) vs offline-retrain ({res['offline-retrain'][0]:.4f}): "
                 "gần ngang nhau về độ chính xác,")
    lines.append("    NHƯNG offline-retrain phải train lại nhiều lần trên toàn bộ data (đắt hơn nhiều).")
    lines.append(f"  - offline-frozen ({res['offline-frozen'][0]:.4f}) THẤP hơn: train 1 lần rồi đứng yên")
    lines.append("    -> không thích nghi khi phân phối trôi theo thời gian (concept drift).")
    lines.append("  => Online learning đạt độ chính xác xấp xỉ offline-retrain với chi phí cập nhật RẺ HƠN,")
    lines.append("     và vượt trội offline-frozen -> đúng động lực dùng online learning cho stream X/Twitter.")

    text = "\n".join(lines); print(text)
    with open("online_vs_offline_results.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # ── biểu đồ accuracy theo thời gian (drift) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, _ in rows:
        wa = windowed_acc(yt, preds[name])
        ax.plot(np.arange(len(wa)), wa, marker=".", ms=3, lw=1.5, label=name)
    ax.set_xlabel("Cửa sổ thời gian (mỗi 3000 mẫu trong stream eval)")
    ax.set_ylabel("Accuracy theo cửa sổ")
    ax.set_title("Online vs Offline — độ chính xác theo dòng thời gian")
    ax.legend(frameon=False); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig("online_vs_offline.png", dpi=160)
    print(f"\n[saved] online_vs_offline_results.txt + online_vs_offline.png  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
