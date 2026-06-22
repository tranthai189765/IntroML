"""
Train PA model on X (Twitter) embedding dataset directly

The embedding dataset has 2947 posts from X/Twitter with precomputed:
- Text embeddings (1024D)
- Image embeddings (1152D)

This script:
1. Loads embedding data
2. Creates synthetic engagement snapshots (since we only have final values)
3. Builds temporal pairs
4. Trains PA model with uncertainty weighting
5. Saves model and metrics

Usage:
  python train_pa_embeddings_dataset.py --output outputs/x_embeddings_model
"""

import argparse
import sys
import os
import time
import pickle
from pathlib import Path

# In được ký tự unicode (σ, ✅, →...) khi stdout bị pipe trên Windows (cp1252)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from src.twitter_pa.training.pipeline import TwitterPAPipeline
from src.twitter_pa.training.temporal_sampler import TemporalSampler
from src.twitter_pa.training.target_builder import TargetBuilder

# Đường dẫn tới embeddings + CSV (layout BTL_ML_20252)
EMB_DIR = os.environ.get("EMB_DIR", os.path.join('..', 'data_v1', 'embeddings'))
CSV_PATH = os.environ.get("CSV_PATH", os.path.join('..', 'data_v1', 'dataset_aligned.csv'))

# Các mốc snapshot THẬT trong dataset_aligned.csv (cột likes_{g}h, ...)
GRID = [0.5, 1, 1.5, 2, 3, 4, 6, 10, 16, 24, 48, 60, 72]
# Metadata tác giả đưa vào feature (khớp FeatureBuilder.AUTHOR_COLS)
AUTHOR_COLS = [
    "author_log_followers", "author_blue_verified", "author_verified",
    "author_followers_per_day", "author_ff_ratio", "author_age_days",
]
_BASE_TS = pd.Timestamp("2026-01-01", tz="UTC")


def _gk(g):
    return str(g).replace(".", "_")


def _as_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def load_real_snapshots(emb_dir=EMB_DIR, csv_path=CSV_PATH):
    """
    Build a LONG temporal dataframe from the REAL per-snapshot columns.

    Khác bản cũ (chỉ lấy *_6h rồi simulate_snapshots() bịa quỹ đạo ngẫu nhiên),
    hàm này dùng đúng snapshot THẬT đã đo tại 0.5/1/1.5/2/3/4/6h:
        mỗi post -> 7 dòng (age_h, likes/comments/reposts/views thật tại mốc đó,
        label_{g}h thật, + metadata tác giả lặp lại).
    crawl_time = base + age_h  -> create_pairs ghép đúng (t -> t+1) theo thời gian thực.

    Returns: (long_df, text_map, image_map, text_dim, image_dim)
    """
    print("[Data] Loading embeddings...")
    ids = np.load(os.path.join(emb_dir, 'ids.npy'), allow_pickle=True).astype(str)
    text_emb = np.load(os.path.join(emb_dir, 'text_emb.npy'))
    image_emb = np.load(os.path.join(emb_dir, 'image_emb_per_post.npy'))
    text_dim, image_dim = text_emb.shape[1], image_emb.shape[1]   # infer, không hardcode
    print(f"  {len(ids)} posts | text {text_emb.shape} | image {image_emb.shape}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Aligned dataset not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={'id': str})
    id_set = set(ids.tolist())
    df = df[df['id'].isin(id_set)].copy()
    print(f"  Aligned CSV rows matched to embeddings: {len(df)}")
    has_author = all(c in df.columns for c in AUTHOR_COLS)
    print(f"  Author metadata columns present: {has_author}")

    rows = []
    for _, r in df.iterrows():
        pid = str(r['id'])
        text = str(r.get('text', '') or '')
        ip = r.get('img_path')
        image_path = ip if (_as_int(r.get('has_image', 0)) == 1
                            and isinstance(ip, str) and ip) else None
        author = {c: (float(r[c]) if c in df.columns and pd.notna(r[c]) else 0.0)
                  for c in AUTHOR_COLS}
        for g in GRID:
            k = _gk(g)
            rows.append({
                'post_id': pid, 'id': pid, 'text': text, 'image_path': image_path,
                'age_h': float(g),
                'crawl_time': _BASE_TS + pd.Timedelta(hours=g),
                'likes': _as_int(r.get(f'likes_{k}h')),
                'comments': _as_int(r.get(f'comments_{k}h')),
                'reposts': _as_int(r.get(f'reposts_{k}h')),
                'views': _as_int(r.get(f'views_{k}h')),
                'label': _as_int(r.get(f'label_{k}h')),     # nhãn THẬT tại mốc g
                **author,
            })
    long_df = pd.DataFrame(rows)
    n_posts = long_df['post_id'].nunique()
    print(f"  Long dataset: {len(long_df)} snapshots ({len(GRID)} per post) "
          f"across {n_posts} posts")
    print(f"  Final-label ({GRID[-1]}h) counts:\n"
          f"{long_df[long_df['age_h'] == GRID[-1]]['label'].value_counts().sort_index().to_string()}\n")

    keep = set(df['id'].tolist())
    text_map = {str(ids[i]): text_emb[i] for i in range(len(ids)) if str(ids[i]) in keep}
    image_map = {str(ids[i]): image_emb[i] for i in range(len(ids)) if str(ids[i]) in keep}
    return long_df, text_map, image_map, text_dim, image_dim


def split_posts(long_df, eval_size: int, seed: int = 42, stratify: bool = True):
    """
    Split by POST (mọi snapshot của 1 post cùng 1 phía -> không leak).

    stratify=True: lấy eval theo TỈ LỆ ĐỀU mỗi lớp (label cuối 6h) -> lớp hiếm
    (Viral 5%) vẫn có mặt đủ trong cả train lẫn eval, ước lượng macro-F1 ổn định hơn.
    """
    from sklearn.model_selection import train_test_split

    # 1 dòng / post kèm label cuối (6h)
    last = (long_df[long_df['age_h'] == GRID[-1]][['post_id', 'label']]
            .drop_duplicates('post_id').reset_index(drop=True))
    posts, labels = last['post_id'].values, last['label'].values
    if eval_size <= 0 or eval_size >= len(posts):
        raise ValueError("eval_size must be positive and smaller than #posts")

    train_ids, eval_ids = train_test_split(
        posts, test_size=eval_size, random_state=seed,
        stratify=labels if stratify else None,
    )
    eval_set = set(eval_ids)
    df_eval = long_df[long_df['post_id'].isin(eval_set)].reset_index(drop=True)
    df_train = long_df[~long_df['post_id'].isin(eval_set)].reset_index(drop=True)

    mode = "stratified theo label" if stratify else "random"
    print(f"[Split] {mode} | Train posts: {len(train_ids)}, Eval posts: {len(eval_ids)}")
    lab = dict(zip(last['post_id'], last['label']))
    tr = pd.Series([lab[p] for p in train_ids]).value_counts(normalize=True).sort_index()
    ev = pd.Series([lab[p] for p in eval_ids]).value_counts(normalize=True).sort_index()
    print("  label %:   " + "  ".join(
        f"L{k}: train={tr.get(k,0)*100:4.1f}% eval={ev.get(k,0)*100:4.1f}%" for k in [0, 1, 2, 3]))
    return df_train, df_eval


def run_kfold(long_df, text_emb_map, image_emb_map, text_dim, image_dim, args):
    """Stratified K-fold CV (split by post on final 6h label). Report ALL metrics."""
    import io, contextlib
    from sklearn.model_selection import StratifiedKFold

    last = (long_df[long_df['age_h'] == GRID[-1]][['post_id', 'label']]
            .drop_duplicates('post_id').reset_index(drop=True))
    posts, labels = last['post_id'].values, last['label'].values
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

    reg_t = ['likes', 'comments', 'reposts', 'views']
    cls_keys = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    A = {k: [] for k in cls_keys}
    R = {t: {'mae': [], 'rmse': []} for t in reg_t}
    BL = {lab: {t: {'mae': [], 'rmse': []} for t in reg_t} for lab in (0, 1, 2, 3)}
    online_acc = []
    oof_all = []

    print(f"\n[K-Fold] {args.kfold}-fold stratified CV (algo={args.algo}, "
          f"side_scale={args.side_scale}, reg={args.reg_c}, cls={args.cls_c}, "
          f"author={not args.no_author}, age={not args.no_age})")
    for fold, (_, ev_idx) in enumerate(skf.split(posts, labels), 1):
        eval_set = set(posts[ev_idx])
        df_tr = long_df[~long_df['post_id'].isin(eval_set)].reset_index(drop=True)
        df_ev = long_df[long_df['post_id'].isin(eval_set)].reset_index(drop=True)
        with contextlib.redirect_stdout(io.StringIO()):
            pipe = train_pa_model(df_tr, text_emb_map, image_emb_map, text_dim, image_dim,
                                  side_scale=args.side_scale,
                                  add_author=not args.no_author, add_age=not args.no_age,
                                  reg_c=args.reg_c, cls_c=args.cls_c,
                                  algo=args.algo, sgd_l2=args.sgd_l2)
            _, m, cls, _, oof = evaluate_temporal_holdout(pipe, df_ev)
        oof_all.append(oof)
        for k in cls_keys:
            A[k].append(cls[k])
        for t in reg_t:
            R[t]['mae'].append(m[t]['mae']); R[t]['rmse'].append(m[t]['rmse'])
        for lab, dd in m.get('by_label', {}).items():
            for t in reg_t:
                BL[lab][t]['mae'].append(dd[t]['mae']); BL[lab][t]['rmse'].append(dd[t]['rmse'])
        online_acc.append(pipe.train_loss_summary.get('online_train_accuracy', float('nan')))
        print(f"  fold {fold}/{args.kfold}: acc={cls['accuracy']:.4f}  f1_macro={cls['f1_macro']:.4f}"
              f"  viewsMAE={m['views']['mae']:.1f}  likesMAE={m['likes']['mae']:.2f}")

    print("\n" + "=" * 64)
    print(f"K-FOLD RESULT — mean ± std over {args.kfold} folds")
    print("=" * 64)
    print("Classification (label 0..3):")
    for k in cls_keys:
        a = np.array(A[k]); print(f"  {k:18}: {a.mean():.4f} ± {a.std():.4f}")
    print("Regression (raw-scale error of NEXT snapshot):")
    for t in reg_t:
        mae, rmse = np.array(R[t]['mae']), np.array(R[t]['rmse'])
        print(f"  {t:10}: MAE {mae.mean():>9.3f} ± {mae.std():<8.3f}  "
              f"RMSE {rmse.mean():>10.2f} ± {rmse.std():.2f}")
    names = {0: 'Low', 1: 'Medium', 2: 'Popular', 3: 'Viral'}
    print("\nRegression MAE/RMSE theo TỪNG LABEL (trung bình các post trong label, mean qua folds):")
    for lab in (0, 1, 2, 3):
        n_any = next((len(BL[lab][t]['mae']) for t in reg_t if BL[lab][t]['mae']), 0)
        if not n_any:
            continue
        print(f"  --- label {lab} ({names[lab]}) ---")
        for t in reg_t:
            mae, rmse = np.array(BL[lab][t]['mae']), np.array(BL[lab][t]['rmse'])
            if len(mae):
                print(f"    {t:10}: MAE {mae.mean():>10.2f}   RMSE {rmse.mean():>11.2f}")
    oa = np.array(online_acc)
    print(f"Online train acc   : {oa.mean():.4f} ± {oa.std():.4f}")
    if oof_all:
        oof_df = pd.concat(oof_all, ignore_index=True)
        out = f"oof_pred_{args.algo}_72h.csv"
        oof_df.to_csv(out, index=False)
        print(f"[oof] saved {len(oof_df)} per-post final-label predictions -> {out}")
    return A, R


def train_pa_model(df_temporal_train, text_emb_map, image_emb_map,
                   text_dim, image_dim, side_scale=40.0,
                   add_author=True, add_age=True, reg_c=0.5, cls_c=8.0,
                   algo="pa", sgd_l2=1e-6):
    """Train online model (PA or SGD) with uncertainty weighting on temporal data."""
    name = "Passive-Aggressive" if algo == "pa" else "online SGD (eta0/sqrt(t)+L2)"
    print(f"\n[Training] Initializing {name} pipeline (text={text_dim}, image={image_dim}, "
          f"author={add_author}, age={add_age}, reg={reg_c}, cls={cls_c}"
          f"{', l2=%g' % sgd_l2 if algo == 'sgd' else ''})...")
    pipeline = TwitterPAPipeline(
        text_backend="tfidf",
        text_dim=text_dim,
        image_dim=image_dim,
        regression_C=reg_c,       # PA: C nhỏ -> regression ổn định | SGD: base LR η₀
        classification_C=cls_c,   # PA: C lớn -> classifier hung hăng  | SGD: base LR η₀
        add_metadata=True,        # has_image (+ hour/dow nếu có)
        add_author=add_author,    # <- metadata tác giả (follower, verified, ...)
        add_age=add_age,          # <- tuổi tuyệt đối của snapshot (timestep)
        side_scale=side_scale,    # boost khối feature phụ vs ~2176-d embedding
        algo=algo,                # <- "pa" | "sgd"
        sgd_l2=sgd_l2,
    )

    print(f"[Training] Setting precomputed embeddings...")
    pipeline.feature_builder.set_text_embeddings(text_emb_map)
    pipeline.feature_builder.set_image_embeddings(image_emb_map)

    print(f"[Training] ONLINE temporal fit (post-by-post, snapshot-by-snapshot)...")
    start = time.time()
    pipeline.fit_temporal_online(df_temporal_train)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s ({len(df_temporal_train)/elapsed:.1f} snapshots/sec)")
    
    return pipeline


def evaluate_temporal_holdout(pipeline, df_temporal_eval):
    """Evaluate the pipeline on held-out temporal posts."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

    print(f"\n[Evaluation] Creating temporal pairs for holdout eval...")
    sampler = TemporalSampler()
    df_t_eval, df_t1_eval = sampler.create_pairs(df_temporal_eval)
    print(f"  Eval pairs: {len(df_t_eval)}")
    
    print(f"[Evaluation] Predicting from current snapshots...")
    results = pipeline.predict(df_t_eval)
    
    # Compute true targets from next snapshots
    Y_reg = pipeline.target_builder.regression_targets(df_t1_eval)
    y_cls = pipeline.target_builder.classification_labels(df_t1_eval)
    
    # Regression metrics
    targets = ['likes', 'comments', 'reposts', 'views']
    metrics = {}
    
    print(f"\n[Results] Regression MAE:")
    for j, target in enumerate(targets):
        pred = results[f'pred_{target}'].values
        true = Y_reg[:, j]
        mae = mean_absolute_error(true, pred)
        rmse = float(np.sqrt(mean_squared_error(true, pred)))
        metrics[target] = {'mae': float(mae), 'rmse': float(rmse)}
        print(f"  {target:10s}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    # ---- per-label regression error (gom cặp eval theo FINAL label của post) ----
    finlab = (df_temporal_eval[df_temporal_eval['age_h'] == GRID[-1]][['post_id', 'label']]
              .drop_duplicates('post_id').set_index('post_id')['label'].to_dict())
    pair_lab = df_t_eval['post_id'].map(finlab).to_numpy()
    by_label = {}
    for lab in (0, 1, 2, 3):
        mask = pair_lab == lab
        if mask.sum() == 0:
            continue
        d = {'n': int(mask.sum())}
        for j, target in enumerate(targets):
            pr = results[f'pred_{target}'].values[mask]; tr = Y_reg[mask, j]
            d[target] = {'mae': float(mean_absolute_error(tr, pr)),
                         'rmse': float(np.sqrt(mean_squared_error(tr, pr)))}
        by_label[lab] = d
    metrics['by_label'] = by_label

    # Classification metrics
    accuracy = accuracy_score(y_cls, results['pred_label'].values)
    macro_f1 = f1_score(y_cls, results['pred_label'].values, average='macro', zero_division=0)
    precision_macro = precision_score(y_cls, results['pred_label'].values, average='macro', zero_division=0)
    recall_macro = recall_score(y_cls, results['pred_label'].values, average='macro', zero_division=0)
    cls_metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(macro_f1),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
    }
    print(f"\n[Results] Classification:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {macro_f1:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro): {recall_macro:.4f}")
    
    # OOF nhãn-CUỐI per-post: dự đoán ở cặp obs=GRID[-2] -> next=GRID[-1] (vd 60h->72h)
    last_obs = GRID[-2]
    mlast = df_t_eval['age_h'].to_numpy() == last_obs
    oof_df = pd.DataFrame({
        'post_id':    df_t_eval['post_id'].to_numpy()[mlast],
        'pred_label': results['pred_label'].to_numpy()[mlast].astype(int),
        'true_label': np.asarray(y_cls)[mlast].astype(int),
    })

    return results, metrics, cls_metrics, len(df_t_eval), oof_df


def save_artifacts(output_dir, pipeline, results, metrics, cls_metrics, df):
    """Save trained model and results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Artifacts] Saving to {output_dir}...")
    
    # Save model
    model_path = output_path / "pa_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  Model: {model_path.name}")
    
    # Save predictions
    pred_path = output_path / "predictions.csv"
    results.to_csv(pred_path, index=False)
    print(f"  Predictions: {pred_path.name} ({len(results)} rows)")

    # Save per-step training loss history (online learning curve)
    train_history = getattr(pipeline, "train_history", [])
    loss_summary = getattr(pipeline, "train_loss_summary", {})
    if train_history:
        log_path = output_path / "training_log.csv"
        pd.DataFrame(train_history).to_csv(log_path, index=False)
        print(f"  Training log: {log_path.name} ({len(train_history)} steps)")
        # Vẽ loss curve
        try:
            from plot_training_loss import plot_loss
            plot_path = plot_loss(str(log_path), str(output_path / "training_loss.png"))
            print(f"  Loss curve: {Path(plot_path).name}")
        except Exception as exc:
            print(f"  [warn] Không vẽ được loss curve: {exc}")

    # Save metrics (gồm cả train loss summary)
    metrics_path = output_path / "metrics.json"
    import json
    metrics_json = {
        'train_loss': loss_summary,
        'regression': {
            target: {k: float(v) if isinstance(v, np.number) else v
                     for k, v in metrics[target].items()}
            for target in metrics.keys()
        },
        'classification': {k: float(v) if isinstance(v, np.number) else v
                          for k, v in cls_metrics.items()},
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics: {metrics_path.name}")
    
    # Save dataset info
    info_path = output_path / "dataset_info.txt"
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"X/Twitter Embedding Dataset - PA Training\n")
        f.write(f"{'='*50}\n")
        n_posts = df['post_id'].nunique() if 'post_id' in df.columns else len(df)
        f.write(f"Posts: {n_posts}\n")
        f.write(f"Temporal snapshots (REAL 0.5-6h): {len(df)}\n")
        f.write(f"Eval temporal pairs: {len(results)}\n")
        f.write(f"\nText embeddings:  precomputed (BGE-M3)\n")
        f.write(f"Image embeddings: precomputed (SigLIP2)\n")
        f.write(f"Feature dimension: {pipeline.feature_builder.feature_dim}D "
                f"(text + image + meta + age + engagement + author)\n")
        f.write(f"\nModel: Passive-Aggressive with Uncertainty Weighting\n")
        f.write(f"- Regression heads: 4 (likes, comments, reposts, views)\n")
        f.write(f"- Classification heads: 4 (Low, Medium, Popular, Viral)\n")
        f.write(f"\nPerformance (held-out eval):\n")
        for target in ['likes', 'comments', 'reposts', 'views']:
            f.write(f"  {target:10s} MAE: {metrics[target]['mae']:.4f}\n")
        f.write(f"  Accuracy: {cls_metrics['accuracy']:.4f}\n")
        if loss_summary:
            f.write(f"\nOnline training - single total loss (uncertainty weighting, Kendall 2018):\n")
            f.write(f"  L_total (avg)     : {loss_summary.get('avg_l_total'):.4f}\n")
            f.write(f"    - L_reg (avg)   : {loss_summary.get('avg_l_reg'):.4f}\n")
            f.write(f"    - L_cls (avg)   : {loss_summary.get('avg_l_cls'):.4f}\n")
            f.write(f"  final sigma_reg   : {loss_summary.get('final_sigma_reg'):.4f}\n")
            f.write(f"  final sigma_cls   : {loss_summary.get('final_sigma_cls'):.4f}\n")
            f.write(f"  online_train_acc  : {loss_summary.get('online_train_accuracy'):.4f}\n")
    print(f"  Info: {info_path.name}")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train PA model on X/Twitter embedding dataset"
    )
    parser.add_argument('--output', default='outputs/x_embeddings_model',
                        help='Output directory')
    parser.add_argument('--eval-size', type=int, default=47,
                        help='Number of held-out posts for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/eval split')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Use only first N posts (for testing)')
    parser.add_argument('--side-scale', type=float, default=40.0,
                        help='Boost factor for side features (meta/age/author/score) vs the '
                             '~2176-d embeddings. Higher reduces embedding noise on the '
                             'classifier; averaged-PA keeps regression stable. ~40 best.')
    parser.add_argument('--no-author', action='store_true',
                        help='Ablation: drop author metadata features')
    parser.add_argument('--no-age', action='store_true',
                        help='Ablation: drop absolute snapshot age feature')
    parser.add_argument('--no-stratify', action='store_true',
                        help='Use random (not label-stratified) train/eval split')
    parser.add_argument('--kfold', type=int, default=0,
                        help='If >1: stratified K-fold CV, report mean±std of ALL metrics')
    parser.add_argument('--algo', choices=['pa', 'sgd'], default='pa',
                        help="Online optimizer: 'pa' (Passive-Aggressive, closed-form step) "
                             "or 'sgd' (subgradient descent, eta=eta0/sqrt(t) + L2). "
                             "Cùng feature/target/objective -> so sánh có kiểm soát.")
    parser.add_argument('--sgd-l2', type=float, default=1e-6,
                        help='L2 weight decay lambda for SGD (ignored when --algo pa)')
    parser.add_argument('--reg-c', type=float, default=None,
                        help='Base step for regression heads. PA: aggressiveness C (default 0.5). '
                             'SGD: base learning rate eta0 (default 0.005).')
    parser.add_argument('--cls-c', type=float, default=None,
                        help='Base step for classifier. PA: aggressiveness C (default 8.0). '
                             'SGD: base learning rate eta0 (default 0.02).')

    args = parser.parse_args()

    # Base-step defaults phụ thuộc optimizer (SGD không chuẩn hóa theo ‖x‖² như PA
    # nên cần LR nhỏ hơn nhiều trên ~2176-d embedding đã z-score).
    if args.reg_c is None:
        args.reg_c = 0.005 if args.algo == 'sgd' else 0.5
    if args.cls_c is None:
        args.cls_c = 0.02 if args.algo == 'sgd' else 8.0
    
    print("=" * 70)
    print("X/TWITTER — PA MODEL TRAINING (EMBEDDING DATASET)")
    print("=" * 70)
    
    # Step 1: Load REAL temporal snapshots (0.5..6h) + author metadata
    long_df, text_emb_map, image_emb_map, text_dim, image_dim = load_real_snapshots()

    # Optionally subsample posts for a quick test run
    if args.sample_size:
        keep_posts = long_df['post_id'].drop_duplicates().head(args.sample_size).tolist()
        long_df = long_df[long_df['post_id'].isin(keep_posts)].reset_index(drop=True)
        text_emb_map = {k: v for k, v in text_emb_map.items() if k in set(keep_posts)}
        image_emb_map = {k: v for k, v in image_emb_map.items() if k in set(keep_posts)}
        print(f"\n  [Subsampled to {len(keep_posts)} posts for testing]")

    # K-fold CV mode: report mean±std of ALL metrics, then stop
    if args.kfold and args.kfold > 1:
        run_kfold(long_df, text_emb_map, image_emb_map, text_dim, image_dim, args)
        print("\n" + "=" * 70)
        print("✅ K-FOLD CV COMPLETE")
        print("=" * 70)
        return

    # Step 2: Split by post (no snapshot leak across train/eval)
    df_temporal_train, df_temporal_eval = split_posts(
        long_df, args.eval_size, seed=args.seed, stratify=not args.no_stratify)

    # Step 3: Train PA model online on the REAL temporal trajectory
    pipeline = train_pa_model(df_temporal_train, text_emb_map, image_emb_map,
                              text_dim, image_dim, side_scale=args.side_scale,
                              add_author=not args.no_author, add_age=not args.no_age,
                              reg_c=args.reg_c, cls_c=args.cls_c,
                              algo=args.algo, sgd_l2=args.sgd_l2)

    # Step 4: Evaluate on held-out posts
    results, metrics, cls_metrics, eval_pairs, _ = evaluate_temporal_holdout(pipeline, df_temporal_eval)
    print(f"\n[Eval] Held-out temporal pairs: {eval_pairs}")

    # Step 5: Save artifacts
    model_path = save_artifacts(args.output, pipeline, results, metrics, cls_metrics, long_df)
    
    print("\n" + "=" * 70)
    print("✅ PA TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"\nNext steps:")
    print(f"  1. Use model: pipeline = pickle.load(open('{model_path}', 'rb'))")
    print(f"  2. Predict: pipeline.predict(new_posts_df)")
    print(f"  3. Online update: pipeline.partial_fit(new_batch)")


if __name__ == '__main__':
    main()
