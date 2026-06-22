"""
src/features/image_embedder.py
Trích xuất image embedding từ ảnh bài đăng X/Twitter.

Pipeline:
  1. Load ResNet18 pretrained (ImageNet) — bỏ FC layer → 512-dim feature vector
  2. Forward pass mỗi ảnh → raw embedding 512-dim
  3. Giảm chiều bằng PCA → n_components (mặc định 32)
  4. Cache kết quả ra disk (tránh re-compute)

Bài không có ảnh → zero vector (feat_img_* = 0).
Kết quả trả về DataFrame [id, feat_img_0 ... feat_img_{n-1}]
để merge với df chính theo cột 'id'.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm


IMG_EMB_COLS_PREFIX = "feat_img_"
N_PCA_COMPONENTS    = 16


def _build_resnet_extractor() -> tuple[nn.Module, transforms.Compose]:
    """ResNet18 với FC thay bằng Identity → output 512-dim."""
    weights = models.ResNet18_Weights.DEFAULT
    model   = models.resnet18(weights=weights)
    model.fc = nn.Identity()   # thay FC(512→1000) bằng passthrough
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])
    return model, tfm


def _extract_raw_embeddings(
    df: pd.DataFrame,
    media_dir: str,
    model: nn.Module,
    tfm: transforms.Compose,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trả về:
      raw_emb  : (N, 512) float32 — embedding ResNet18 (0 nếu không có ảnh)
      has_emb  : (N,) bool        — True nếu ảnh tồn tại và được đọc thành công
    """
    N      = len(df)
    raw    = np.zeros((N, 512), dtype=np.float32)
    has    = np.zeros(N, dtype=bool)
    ids    = df["id"].tolist()
    paths  = df["img_path"].tolist()

    for i, (pid, img_path) in enumerate(tqdm(
        zip(ids, paths), total=N,
        desc="  ResNet18 embedding", ncols=70,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    )):
        if not img_path or (isinstance(img_path, float) and np.isnan(img_path)):
            continue
        fname = os.path.basename(str(img_path))
        full  = os.path.join(media_dir, fname)
        if not os.path.exists(full):
            continue
        try:
            img = Image.open(full).convert("RGB")
            x   = tfm(img).unsqueeze(0)        # (1, 3, 224, 224)
            with torch.no_grad():
                emb = model(x).numpy()          # (1, 512)
            raw[i] = emb[0]
            has[i] = True
        except Exception:
            pass

    return raw, has


def build_image_embeddings(
    df:           pd.DataFrame,
    media_dir:    str,
    cache_path:   str,
    n_components: int = N_PCA_COMPONENTS,
    fit_df:       pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Trả về DataFrame với cột 'id' và n_components cột 'feat_img_*'.
    Kết quả được cache tại cache_path để chạy lần sau không cần tính lại.

    Args:
        df          : DataFrame cần transform (cần cột 'id' và 'img_path')
        media_dir   : thư mục chứa file ảnh .jpg
        cache_path  : đường dẫn .npz để lưu cache
        n_components: số chiều PCA
        fit_df      : nếu không None, chỉ fit PCA trên subset này (tránh leakage
                      khi dùng held-out test). Mặc định None = fit trên toàn bộ df.
    """
    img_cols = [f"{IMG_EMB_COLS_PREFIX}{i}" for i in range(n_components)]

    # ── Kiểm tra cache PCA (chỉ dùng khi fit_df=None) ───────────────────────
    if fit_df is None and os.path.exists(cache_path):
        cache  = np.load(cache_path)
        embs   = cache["embs"]          # (N, cached_components)
        if embs.shape[1] == n_components:
            print(f"      [image_embedder] Loading cached embeddings from {cache_path}")
            emb_df = pd.DataFrame(embs, columns=img_cols)
            emb_df.insert(0, "id", cache["ids"].tolist())
            return emb_df
        # Cache lệch số chiều (n_components đã đổi) → tính lại để tránh mismatch
        print(f"      [image_embedder] Cache has {embs.shape[1]} dims ≠ "
              f"n_components={n_components} → recomputing")

    # ── Trích xuất / Load Raw Embeddings ──────────────────────────────────────
    raw_cache_path = cache_path.replace(".npz", "_raw.npz")
    if os.path.exists(raw_cache_path):
        print(f"      [image_embedder] Loading raw ResNet18 embeddings from cache: {raw_cache_path}")
        raw_cache = np.load(raw_cache_path)
        raw_emb = raw_cache["raw_emb"]
        has_emb = raw_cache["has_emb"]
    else:
        print("      [image_embedder] Loading ResNet18 (pretrained ImageNet) ...")
        model, tfm = _build_resnet_extractor()
        raw_emb, has_emb = _extract_raw_embeddings(df, media_dir, model, tfm)
        os.makedirs(os.path.dirname(raw_cache_path), exist_ok=True)
        np.savez(raw_cache_path, raw_emb=raw_emb, has_emb=has_emb)
        print(f"      [image_embedder] Saved raw embeddings cache → {raw_cache_path}")

    n_found = has_emb.sum()
    print(f"      [image_embedder] Embedded {n_found}/{len(df)} images")

    # ── PCA 512 → n_components ────────────────────────────────────────────────
    # Nếu fit_df được cung cấp: chỉ fit PCA trên đó (train only), sau đó transform df
    if fit_df is not None:
        # Lọc các index của fit_df trong df
        df_id_to_idx = {pid: idx for idx, pid in enumerate(df["id"])}
        fit_indices = [df_id_to_idx[pid] for pid in fit_df["id"] if pid in df_id_to_idx]
        
        fit_raw = raw_emb[fit_indices]
        fit_has = has_emb[fit_indices]
        n_fit = fit_has.sum()
        
        print(f"      [image_embedder] Fitting PCA {512}→{n_components} "
              f"on {n_fit} train images (leak-free) ...")
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(fit_raw[fit_has])
    else:
        print(f"      [image_embedder] Fitting PCA {512}→{n_components} "
              f"on {n_found} image embeddings ...")
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(raw_emb[has_emb])

    explained = pca.explained_variance_ratio_.sum()
    print(f"      [image_embedder] PCA variance explained: {explained*100:.1f}%")

    reduced = np.zeros((len(df), n_components), dtype=np.float32)
    reduced[has_emb] = pca.transform(raw_emb[has_emb])

    # ── Lưu cache PCA (chỉ khi fit trên toàn df — cache với fit_df khác nhau sẽ sai) ──
    if fit_df is None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, ids=df["id"].tolist(), embs=reduced)
        print(f"      [image_embedder] Saved PCA cache → {cache_path}")

    emb_df = pd.DataFrame(reduced, columns=img_cols)
    emb_df.insert(0, "id", df["id"].tolist())
    return emb_df


def get_img_feature_cols(n_components: int = N_PCA_COMPONENTS) -> list[str]:
    return [f"{IMG_EMB_COLS_PREFIX}{i}" for i in range(n_components)]
