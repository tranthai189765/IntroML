"""
Embed data/dataset_aligned.csv on the server (GPU recommended):
  - text  -> BGE-M3 dense  (1024-d)
  - image -> SigLIP2        (each image file gets its own embedding)

Saves embeddings as .npy/.npz (NOT inside the CSV — far smaller, exact, fast),
aligned to the CSV by post `id`:

  data/embeddings/
    ids.npy                  (N,)        post ids in CSV row order (alignment key)
    text_emb.npy             (N, 1024)   BGE-M3 dense, L2-normalized
    image_emb_per_image.npz  files[],emb (M, D)  one row per image file
    image_emb_per_post.npy   (N, D)      mean-pooled over a post's images (0 if none)
    has_image.npy            (N,) bool
    dataset_embeddings.parquet           metadata + labels + text_emb + image_emb (1 file to share)

Usage:  python embed_dataset.py
"""
import pathlib
import numpy as np
import pandas as pd
import torch
from PIL import Image

DATA = pathlib.Path("data")
CSV = DATA / "dataset_aligned.csv"
MEDIA = DATA / "media"
OUT = DATA / "embeddings"; OUT.mkdir(parents=True, exist_ok=True)

SIGLIP_MODEL = "google/siglip2-base-patch16-224"   # -> so400m-patch16-384 for higher accuracy
BGE_MODEL = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_BATCH = 64
MAX_IMGS_PER_POST = 6


def l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    df = pd.read_csv(CSV)
    ids = df["id"].astype(str).tolist()
    np.save(OUT / "ids.npy", np.array(ids))
    print(f"[data] {len(ids)} posts | device={DEVICE}")

    # ---------------- TEXT: BGE-M3 (dense 1024-d) ----------------
    from FlagEmbedding import BGEM3FlagModel
    print("[text] loading BGE-M3 ...")
    tm = BGEM3FlagModel(BGE_MODEL, use_fp16=(DEVICE == "cuda"))
    texts = df["text"].fillna("").astype(str).tolist()
    text_emb = tm.encode(texts, batch_size=32, max_length=512)["dense_vecs"]
    text_emb = l2(text_emb)
    np.save(OUT / "text_emb.npy", text_emb)
    print(f"[text] text_emb {text_emb.shape} saved")

    # ---------------- IMAGE: SigLIP2 (per image file) ----------------
    from transformers import AutoModel, AutoProcessor
    print(f"[image] loading {SIGLIP_MODEL} ...")
    sig = AutoModel.from_pretrained(SIGLIP_MODEL).to(DEVICE).eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_MODEL)

    files = sorted(p.name for p in MEDIA.glob("*.jpg"))
    kept_files, vecs = [], []
    for i in range(0, len(files), IMG_BATCH):
        chunk, imgs = files[i:i + IMG_BATCH], []
        ok = []
        for fn in chunk:
            try:
                imgs.append(Image.open(MEDIA / fn).convert("RGB")); ok.append(fn)
            except Exception:
                pass
        if not imgs:
            continue
        inp = proc(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = sig.get_image_features(**inp).float().cpu().numpy()
        kept_files.extend(ok); vecs.append(feats)
        if i % (IMG_BATCH * 10) == 0:
            print(f"  image {i+len(chunk)}/{len(files)}")
    img_emb = l2(np.vstack(vecs)) if vecs else np.zeros((0, 1), np.float32)
    D = img_emb.shape[1]
    np.savez(OUT / "image_emb_per_image.npz", files=np.array(kept_files), emb=img_emb)
    print(f"[image] per-image {img_emb.shape} saved ({len(kept_files)} files)")

    # ---- per-post image embedding = mean-pool of that post's images ----
    idx = {fn: v for fn, v in zip(kept_files, img_emb)}
    per_post = np.zeros((len(ids), D), dtype=np.float32)
    has_img = np.zeros(len(ids), dtype=bool)
    for r, pid in enumerate(ids):
        vs = [idx[f"{pid}_{k}.jpg"] for k in range(MAX_IMGS_PER_POST)
              if f"{pid}_{k}.jpg" in idx]
        if vs:
            per_post[r] = l2(np.mean(vs, axis=0, keepdims=True))[0]
            has_img[r] = True
    np.save(OUT / "image_emb_per_post.npy", per_post)
    np.save(OUT / "has_image.npy", has_img)
    print(f"[image] per-post {per_post.shape} | with image: {int(has_img.sum())}/{len(ids)}")

    # ---------------- one shareable parquet (metadata + labels + embeddings) ----------------
    try:
        out = df.copy()
        out["text_emb"] = list(text_emb.astype(np.float32))
        out["image_emb"] = list(per_post.astype(np.float32))
        out["has_image_emb"] = has_img
        out.to_parquet(OUT / "dataset_embeddings.parquet", index=False)
        print(f"[parquet] dataset_embeddings.parquet saved")
    except Exception as e:
        print(f"[parquet] skipped ({e}); use the .npy files instead")

    print("\nDONE ->", OUT)


if __name__ == "__main__":
    main()
