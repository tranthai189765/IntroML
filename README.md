# IntroML — X (Twitter) Online-Learning Data Pipeline

Thu thập dữ liệu post **mới (0–1h tuổi) + đang trong topic thịnh hành** trên X
(Twitter), lọc spam, tải ảnh, và **theo dõi engagement (views/likes/shares) theo
thời gian** trong ~8h đầu của mỗi post. Đầu ra là một CSDL SQLite sẵn sàng để
huấn luyện mô hình **online-learning đa phương thức (text + ảnh → dự đoán
engagement tương lai)**.

Dữ liệu lấy qua REST API của [twitterapi.io](https://twitterapi.io) (header
`x-api-key`, không cần OAuth).

---

## 1. Vì sao thiết kế như vậy (tóm tắt cơ sở đo đạc)

| Quyết định | Lý do (đo thật từ dữ liệu) |
|---|---|
| **Intake post tuổi 0–1h** | Post chụp lúc ≤1h vẫn còn tăng **+56–92% like** về sau → snapshot mang nhiều biến động để học. Post intake >2–3h đã bão hòa >70% (label "chết"). |
| **Không lọc `min_faves` lúc intake** | Ở tuổi 0–1h chưa biết post nào sẽ viral — **độ viral chính là label cần dự đoán**. |
| **Toán tử `since_time:<epoch>`** | Chỉ lấy post trong 1h qua → không trả tiền cho post cũ. |
| **Snapshot theo lịch `[1,2,3,4,6,8]h`** | Dày lúc đầu vì engagement biến động mạnh ở vài giờ đầu (đo được: +73% giờ đầu → ~+8% lúc ~5h). |
| **Retire sau 8h** | Post đã bão hòa quanh ~5–6h → ngừng re-fetch để tiết kiệm chi phí + chín nhanh. |

---

## 2. Luồng crawl dữ liệu (data flow)

```
                          ┌─────────────────────────────────────────────┐
                          │            MỖI CYCLE (mặc định 1h)            │
                          └─────────────────────────────────────────────┘

  [1] INTAKE  ──────────────────────────────────────────────────────────────────
      cho mỗi WOEID (vùng: world/US/JP/BR/IN ...):
          GET /twitter/trends?woeid=..              → danh sách topic thịnh hành
          cho mỗi topic (top 10):
              GET /twitter/tweet/advanced_search
                  query = "<topic> since_time:<now-1h>"  queryType=Top   (2 trang)
                  → chỉ post tuổi 0–1h, sắp theo engagement
      gộp candidates → DEDUP theo tweet id (bỏ post đã có trong DB)

  [2] SPAM FILTER  ─────────────────────────────────────────────────────────────
      loại: dup_text | template | promo(cashtag) | author_flood | author_db
      (in breakdown số lượng theo từng lý do)

  [3] TẢI ẢNH  ─────────────────────────────────────────────────────────────────
      lấy ảnh từ extendedEntities (photo → ảnh; video → thumbnail)
      tải song song về  data/media/<post_id>_<n>.jpg   (KHÔNG tốn credit API)

  [4] LƯU  ─────────────────────────────────────────────────────────────────────
      INSERT posts(...)            (1 dòng/post)
      INSERT snapshots(...) t0     (điểm dữ liệu đầu tiên, tuổi lúc intake)

  ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

  [5] SNAPSHOT (mỗi cycle, trước khi intake)  ──────────────────────────────────
      chọn post chưa retire mà tuổi ≥ mốc lịch kế tiếp
          GET /twitter/tweets?tweet_ids=...   (batch 50 id/call)
          INSERT snapshots(...)   (ghi views/likes/shares tại tuổi hiện tại)
          tiến con trỏ lịch; nếu qua mốc cuối hoặc tuổi ≥8h → RETIRE post
```

Mỗi post vì thế có **một chuỗi snapshot theo thời gian** = nhãn (label) cho bài
toán dự đoán engagement.

---

## 3. Cài đặt

```bash
pip install -r requirements.txt          # chỉ cần: requests
# tạo file .env chứa API key (KHÔNG commit file này):
echo "API_KEY=your_twitterapi_io_key" > .env
```

> ⚠️ **Bảo mật:** `.env` (API key) và `data/` (DB + ảnh) đã được liệt kê trong
> `.gitignore` — không bao giờ đẩy lên GitHub.

---

## 4. Các lệnh chạy

```bash
python x_pipeline.py init                                   # tạo data/x_pipeline.db
python x_pipeline.py intake   --target 5000 --budget 2.0    # 1 lần intake post mới
python x_pipeline.py snapshot                               # re-snapshot post tới hạn
python x_pipeline.py snapshot --force                       # re-snapshot TẤT CẢ (debug)
python x_pipeline.py stats                                  # tóm tắt DB + vài trajectory

# chạy liên tục (server): mỗi cycle = snapshot tới hạn + intake mới, tới khi đủ
# target, rồi tiếp tục snapshot cho đến khi mọi post retire. Có cap chi phí (USD).
python x_pipeline.py run --target 5000 --interval 3600 --budget 8.0
```

| Lệnh | Chức năng |
|---|---|
| `init` | Tạo CSDL + bảng (idempotent) |
| `intake` | Crawl post fresh 0–1h, lọc spam, tải ảnh, lưu + snapshot t0 |
| `snapshot` | Re-fetch metrics các post đến hạn theo lịch tuổi |
| `snapshot --force` | Re-fetch toàn bộ post active ngay (để test/đặt thêm điểm) |
| `run` | Vòng lặp liên tục cho server (resume được) |
| `stats` | In số post/ảnh/snapshot + ví dụ trajectory |

**Tham số chỉnh nhanh** (đầu file [x_pipeline.py](x_pipeline.py)):
`WOEIDS`, `TRENDS_PER_WOEID`, `PAGES_PER_TOPIC`, `FRESH_WINDOW_H`,
`SNAPSHOT_AGES_H`, `MAX_TRACK_H`, `SNAPSHOT_BATCH`, và các ngưỡng spam.

---

## 5. Dữ liệu được lưu ở đâu

```
data/
├── x_pipeline.db          # SQLite — toàn bộ post + chuỗi snapshot
└── media/
    └── <post_id>_<n>.jpg  # ảnh đã tải (n = chỉ số ảnh trong post, tối đa 4)
```

### Bảng `posts` (1 dòng / post)
| Cột | Ý nghĩa |
|---|---|
| `id` | tweet id (khóa chính) |
| `author` | username người đăng |
| `text` | nội dung post |
| `lang` | ngôn ngữ |
| `created_epoch` | thời điểm post được tạo (Unix) |
| `intake_epoch` | thời điểm crawl vào DB (Unix) |
| `intake_age_h` | tuổi post lúc intake (giờ) |
| `has_image` / `has_video` | 1/0 |
| `image_urls` / `video_urls` | JSON list URL gốc |
| `media_paths` | JSON list đường dẫn ảnh local đã tải |
| `url` | link post |
| `next_snap_idx` | con trỏ mốc snapshot kế tiếp |
| `retired` | 1 = đã ngừng theo dõi |

### Bảng `snapshots` (nhiều dòng / post = chuỗi thời gian)
| Cột | Ý nghĩa |
|---|---|
| `post_id` | khóa ngoại → posts.id |
| `snap_epoch` | thời điểm chụp metrics (Unix) |
| `age_h` | tuổi post tại thời điểm chụp (giờ) |
| `likes`, `views`, `retweets`, `replies`, `quotes`, `bookmarks` | metrics |

> Khóa chính `(post_id, snap_epoch)` → chống ghi trùng, an toàn khi resume.

### Truy vấn lấy dữ liệu train (ví dụ: features t0 → label cuối)
```sql
SELECT p.id, p.text, p.media_paths,
       s0.likes  AS likes_t0,    s0.views  AS views_t0,
       sL.likes  AS likes_final, sL.views  AS views_final
FROM posts p
JOIN snapshots s0 ON s0.post_id=p.id AND s0.snap_epoch=p.intake_epoch
JOIN snapshots sL ON sL.post_id=p.id
WHERE sL.age_h = (SELECT MAX(age_h) FROM snapshots WHERE post_id=p.id);
```

---

## 6. Lọc spam (luôn bật)

Áp dụng tại bước intake, in breakdown theo từng lý do mỗi pass:

| Lý do | Mô tả |
|---|---|
| `dup_text` | text chuẩn hóa trùng hệt ≥3 lần |
| `template` | cùng "khung" (text bỏ #/$tag & số) ≥4 lần — bắt campaign đổi tag |
| `promo` | ≥3 cashtag `$XXX`, hoặc text ngắn nhồi tag — bơm cổ phiếu |
| `author_flood` | 1 account xuất hiện ≥5 lần trong pass (giữ post like cao nhất) |
| `author_db` | account đã có ≥8 post trong DB (flood xuyên pass) |

Chỉnh ngưỡng ở đầu file `x_pipeline.py`.

---

## 7. Chi phí & ngân sách

- Giá: **100k credit = $1**; tweet **$0.15/1k** (15 credit/tweet); tối thiểu
  $0.00015/request.
- `--budget USD` là **trần cứng**/lần chạy (ước lượng local, dừng trước khi vượt).
  Sau mỗi lần chạy in `[cost]` = số credit thật đã tiêu + số dư còn lại.
- Mỗi post ≈ 1 fetch intake + ~6 fetch snapshot ≈ **$0.0012/post**:

| Dataset | ≈ Chi phí |
|---|---|
| 5.000 post | ~$6 |
| 10.000 post | ~$12 |
| 20.000 post | ~$24 |

**Lưu ý nguồn cung:** ~1.900 post fresh/giờ từ 5 vùng → để đủ 5k+ cần chạy `run`
lặp theo giờ (mỗi giờ có post mới) hoặc thêm vùng vào `WOEIDS`.

**Lưu ý text+image:** chỉ ~39% post fresh có ảnh → ~61% là text-only. Pipeline
giữ cả hai; mô hình tự xử lý trường hợp thiếu ảnh.

---

## 8. Triển khai server (chạy không ngừng)

`run` resume được (SQLite + dedup theo id) → khởi động lại lúc nào cũng an toàn.

```bash
# Linux (nohup) — nhớ -u để log ra real-time
nohup python -u x_pipeline.py run --target 20000 --interval 3600 --budget 40 \
      >> pipeline.log 2>&1 &

# hoặc cron: 1 cycle/giờ
0 * * * * cd /path/repo && python x_pipeline.py snapshot --budget 5 && \
          python x_pipeline.py intake --target 20000 --budget 5
```

Giữ `.env` trên server, **không commit**.

---

## 9. Script phân tích (đã dùng để chọn tham số thiết kế)

| File | Mục đích |
|---|---|
| [crawl_trending.py](crawl_trending.py) | Crawl thử N post Top của 1 topic → JSON/CSV |
| [build_csv.py](build_csv.py) | Dựng lại CSV (kèm cột ảnh/video) từ JSON |
| [freshness.py](freshness.py) | Phân bố tuổi của post Top so với hiện tại |
| [freshness_ratio.py](freshness_ratio.py) | Tỷ lệ post 0–1h trên nhiều topic |
| [analyze_figures.py](analyze_figures.py) | Dựng figure/bảng chất lượng data từ DB → `data/figures/` (phân phối, ngưỡng, đường bão hòa, bảng 1/2/3) |

---

## 10. Cấu trúc repo

```
.
├── x_pipeline.py        # PIPELINE CHÍNH (intake/snapshot/run/stats)
├── requirements.txt
├── README.md            # file này
├── .gitignore           # loại .env và data/
├── crawl_trending.py    # script phân tích/khảo sát
├── build_csv.py
├── freshness.py
└── freshness_ratio.py
# (KHÔNG đẩy lên git: .env, data/x_pipeline.db, data/media/)
```
