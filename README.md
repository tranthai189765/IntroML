# IntroML — Dự đoán độ phổ biến (Virality) trên Mạng xã hội bằng Học trực tuyến

Dự đoán độ **viral** của bài đăng trên **X/Twitter** và **Reddit** theo thời gian thực bằng
**Học trực tuyến (Online Learning)** trên đặc trưng **đa phương thức**: văn bản (BGE‑M3) +
hình ảnh (SigLIP 2) + tác giả + các chỉ số tương tác.

Ba thuật toán học trực tuyến được so sánh trên **cùng một bộ đặc trưng**:

| Thuật toán | Họ mô hình | Vai trò |
|---|---|---|
| **Passive‑Aggressive (PA)** | Tuyến tính, cập nhật theo lề | Phân loại + hồi quy |
| **online SGD** | Tuyến tính, gradient ngẫu nhiên | Phân loại + hồi quy |
| **Adaptive Random Forest (ARF)** | Tập hợp cây + phát hiện trôi (ADWIN) | Phân loại + hồi quy |

> **Dữ liệu** (CSV, embeddings, ảnh) **không nằm trong repo**. Tải tại Google Drive:
> **https://drive.google.com/drive/folders/1r6nTkj1fvEfbtamp9AvV4Wj2M3cxmnHB**

---

## Mục lục
1. [Tổng quan bài toán](#1-tổng-quan-bài-toán)
2. [Cấu trúc repository](#2-cấu-trúc-repository)
3. [Dữ liệu & cách đặt thư mục](#3-dữ-liệu--cách-đặt-thư-mục)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Bước 1 — Trích xuất đặc trưng (embedding)](#5-bước-1--trích-xuất-đặc-trưng-embedding)
6. [Bước 2 — Chạy thuật toán theo từng loại dữ liệu](#6-bước-2--chạy-thuật-toán-theo-từng-loại-dữ-liệu)
7. [Bước 3 — Đánh giá](#7-bước-3--đánh-giá)
8. [Pipeline thu thập & xây dựng dữ liệu (tùy chọn)](#8-pipeline-thu-thập--xây-dựng-dữ-liệu-tùy-chọn)
9. [Tóm tắt kết quả](#9-tóm-tắt-kết-quả)

---

## 1. Tổng quan bài toán

Mỗi bài đăng được theo dõi qua **13 mốc thời gian** (snapshot) từ `0.5h` đến `72h`. Mục tiêu:

- **Phân loại 4 lớp độ viral**: `0 Low` (đáy 50%) · `1 Medium` (50–80%) · `2 Popular` (80–95%) · `3 Viral` (top 5%).
- **Hồi quy** các chỉ số tương tác tại snapshot kế tiếp.
- **Xếp hạng chủ đề** (topic ranking) và **so sánh Online vs Offline learning**.

Mỗi mẫu `(bài đăng, snapshot)` được biểu diễn bởi một vector **≈ 3.234 chiều**:

| Khối đặc trưng | Mô hình / phương pháp | Số chiều |
|---|---|---|
| Văn bản | BGE‑M3 | 1.024 |
| Hình ảnh | SigLIP 2 (`so400m-patch16-384`) | 1.152 |
| Tác giả | BGE‑M3 (tên tài khoản) | 1.024 |
| Số (tương tác / thời gian / siêu dữ liệu) | thủ công | ≈ 34 |

---

## 2. Cấu trúc repository

```
IntroML/
├── README.md  ·  requirements.txt
├── embed_dataset.py              # Tải BGE-M3 + SigLIP2, embed text/ảnh/tác giả -> .npy
├── x_pipeline.py                 # crawl X/Twitter theo 13 snapshot
├── crawl_trending.py · build_csv.py · filter_consistent.py · freshness*.py · analyze_figures.py
│
├── algorithms/
│   ├── pa_sgd/                   # Passive-Aggressive & online SGD (cùng pipeline)
│   │   ├── train_pa_embeddings_dataset.py   # X/Twitter (train + eval K-fold)
│   │   ├── run_reddit_data.py               # Reddit
│   │   └── src/                             # thư viện pipeline (feature builder, PA core, ...)
│   └── arf/                      # Adaptive Random Forest (river)
│       ├── run_arf_72h.py                   # X/Twitter (prequential + eval)
│       └── src/
│
├── evaluation/
│   ├── topic_ranking.py          # xếp hạng chủ đề + so khớp ranking dự đoán/thực tế
│   ├── online_vs_offline.py      # so sánh online vs 2 baseline offline
│   ├── report_figures.py         # sinh hình cho báo cáo
│   └── simulate_results.py
│
└── data_pipeline/                # script bổ sung (thu thập + xây dựng dữ liệu)
    ├── crawl_mature.py           # crawl post viral đã chín (>3 ngày)
    ├── add_author_meta.py        # bổ sung metadata tác giả
    ├── backfill_mature.py        # suy ngược các mốc sớm
    ├── simulate_reddit.py        # sinh dữ liệu Reddit từ động học X/Twitter
    └── engagement_dynamics.py    # phân tích động học tương tác
```

---

## 3. Dữ liệu & cách đặt thư mục

Tải dữ liệu từ **[Google Drive](https://drive.google.com/drive/folders/1r6nTkj1fvEfbtamp9AvV4Wj2M3cxmnHB)** và đặt vào thư mục `data/` ở gốc repo:

```
data/
├── dataset_72h.csv            # X/Twitter ~10k bài (đủ 13 snapshot)
├── 50k_dataset_72h.csv        # X/Twitter ~50k bài (kèm 11 cột metadata tác giả)
├── dataset_reddit_72h.csv     # Reddit (sinh từ động học X/Twitter)
├── embeddings_72h/            # embeddings .npy (đã căn theo id của CSV)
│   ├── ids.npy  text_emb.npy  author_emb.npy
│   ├── image_emb_per_post.npy  image_emb_per_image.npz  has_image.npy
│   └── dataset_embeddings.parquet
└── media/                     # ảnh đặt tên {id}_{k}.jpg
```

> Mọi đường dẫn đều cấu hình được qua biến môi trường `CSV_PATH` và `EMB_DIR` (xem các lệnh bên dưới).

---

## 4. Cài đặt môi trường

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **GPU (CUDA) được khuyến nghị** cho bước embedding (BGE‑M3 + SigLIP 2).
- Bước chạy thuật toán PA/SGD/ARF chạy tốt trên **CPU**.

---

## 5. Bước 1 — Trích xuất đặc trưng (embedding)

`embedding/embed_dataset.py` **tự động tải model** `BAAI/bge-m3` và `google/siglip2-so400m-patch16-384`
từ HuggingFace, rồi sinh embeddings cho **văn bản + hình ảnh + tên tác giả**.

```bash
# Đặt CSV cần embed tại data/dataset_aligned.csv và ảnh tại data/media/
python embed_dataset.py
# -> kết quả lưu tại data/embeddings/   (ids.npy, text_emb.npy, author_emb.npy,
#    image_emb_per_post.npy, has_image.npy, dataset_embeddings.parquet)
```

> Nếu đã tải sẵn `embeddings_72h/` từ Google Drive thì **bỏ qua bước này**.

---

## 6. Bước 2 — Chạy thuật toán theo từng loại dữ liệu

### 6.1 X/Twitter — Passive‑Aggressive & online SGD

```bash
# Passive-Aggressive (5-fold CV: phân loại + hồi quy + breakdown theo nhãn)
CSV_PATH=data/dataset_72h.csv EMB_DIR=data/embeddings_72h \
  python algorithms/pa_sgd/train_pa_embeddings_dataset.py --kfold 5 --algo pa

# online SGD (cùng pipeline, chỉ đổi --algo)
CSV_PATH=data/dataset_72h.csv EMB_DIR=data/embeddings_72h \
  python algorithms/pa_sgd/train_pa_embeddings_dataset.py --kfold 5 --algo sgd
```

> Dùng `50k_dataset_72h.csv` bằng cách đổi `CSV_PATH` (và `EMB_DIR` tương ứng).
> Mỗi lần chạy sinh `oof_pred_<algo>_72h.csv` (dự đoán nhãn cuối per‑post) phục vụ topic ranking.

### 6.2 X/Twitter — Adaptive Random Forest

```bash
CSV_PATH=data/dataset_72h.csv EMB_DIR=data/embeddings_72h \
  python algorithms/arf/run_arf_72h.py
# tùy chọn: N_POSTS=3500 (chạy subset cho nhanh) · WARMUP, PCA_IMG/PCA_TXT/PCA_AUTH
```

### 6.3 Reddit (upvotes / downvotes / views / comments)

```bash
# (a) sinh dữ liệu Reddit từ động học X/Twitter
python data_pipeline/simulate_reddit.py        # -> data/dataset_reddit_72h.csv

# (b) chạy PA / online SGD trên Reddit
python algorithms/pa_sgd/run_reddit_data.py
```

| Dữ liệu | PA | online SGD | ARF |
|---|---|---|---|
| **X/Twitter** | `train_pa_embeddings_dataset.py --algo pa` | `--algo sgd` | `run_arf_72h.py` |
| **Reddit** | `run_reddit_data.py` | `run_reddit_data.py` | `run_arf_72h.py` (trỏ `CSV_PATH` tới CSV Reddit) |

---

## 7. Bước 3 — Đánh giá

```bash
# Xếp hạng chủ đề: gom điểm (Viral=5, Popular=3, Medium=1, Low=0) theo topic,
# so khớp ranking dự đoán vs thực tế (Spearman ρ, tier-accuracy, Top-3 overlap)
python evaluation/topic_ranking.py

# Online vs Offline: online (PA/SGD/ARF) vs offline-retrain & offline-frozen
python evaluation/online_vs_offline.py

# Sinh hình minh hoạ cho báo cáo
python evaluation/report_figures.py
```

---

## 8. Pipeline thu thập & xây dựng dữ liệu (tùy chọn)

Toàn bộ quá trình tạo dữ liệu (không bắt buộc chạy lại — dữ liệu đã có trên Drive):

```bash
python x_pipeline.py run                       # crawl X/Twitter theo 13 snapshot
python data_pipeline/crawl_mature.py          # crawl post viral đã chín (>3 ngày)
python data_pipeline/backfill_mature.py       # suy ngược 12 mốc sớm
python data_pipeline/add_author_meta.py data/50k_dataset_72h.csv   # bổ sung metadata tác giả
```

> **Bảo mật:** mọi script đọc API key từ biến môi trường / file `.env` — **không** hard‑code, **không** commit `.env`.

---

## 9. Tóm tắt kết quả

*(Số liệu minh hoạ trong báo cáo; xem chi tiết ở chương Kết quả.)*

**Phân loại độ viral (X/Twitter):**

| Thuật toán | Accuracy | F1‑macro |
|---|---|---|
| PA | 0.885 | 0.771 |
| online SGD | 0.883 | 0.652 |
| **ARF** | **0.948** | **0.926** |

**Online vs Offline (per‑post):**

| Phương pháp | Chế độ | Accuracy |
|---|---|---|
| online‑ARF | Online | **0.905** |
| offline‑retrain | Offline (retrain) | 0.896 |
| online‑PA | Online | 0.884 |
| offline‑frozen | Offline (đóng băng) | 0.828 |

→ **ARF** dẫn đầu về phân loại; **online learning** đạt độ chính xác xấp xỉ *offline‑retrain* nhưng chi phí cập nhật **rẻ hơn nhiều**, và vượt trội *offline‑frozen* (vốn suy giảm theo thời gian do trôi dữ liệu).

---

<sub>Đồ án môn Nhập môn Học máy (IntroML). Dữ liệu lưu trên Google Drive, mã nguồn công khai tại repo này.</sub>
