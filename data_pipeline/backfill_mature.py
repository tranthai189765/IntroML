# -*- coding: utf-8 -*-
"""
backfill_mature.py — suy NGƯỢC 12 mốc sớm cho các post viral đã chín (crawl single-shot).

Input : data_v1/mature_raw.jsonl  (id, author, text, lang, likes/views/comments/reposts hiện tại,
                                    url, img_paths)  -> giá trị hiện tại = mốc 72h THẬT
        data_v1/dataset_72h.csv    (10k post đầy đủ) -> nguồn ĐỘNG LỰC (donor-shape)
Output: data_v1/dataset_mature_72h.csv  (cùng format dataset_72h)
        data_v1/dataset_combined_72h.csv (gộp 10k cũ + mới)

Cách suy ngược:
  - Với mỗi post mới: tính score_final & gán label theo NGƯỠNG phân vị của 10k cũ.
  - Chọn 1 donor CÙNG label trong 10k, lấy đường cong tỉ lệ r_m(t)=value_m(t)/value_m(72h).
  - value_m(t) = F_m * r_m(t) * nhiễu nhỏ, ép ≤ F_m, đơn điệu không giảm, mốc 72h = F_m thật.
  - score_t / label_t tính lại theo ngưỡng phân vị trong-mốc của 10k cũ.
"""
import json, os
import numpy as np
import pandas as pd

SRC_CSV  = "data_v1/dataset_72h.csv"
RAW      = "data_v1/mature_raw.jsonl"
OUT_NEW  = "data_v1/dataset_mature_72h.csv"
OUT_COMB = "data_v1/dataset_combined_72h.csv"
GRID = [0.5, 1, 1.5, 2, 3, 4, 6, 10, 16, 24, 48, 60, 72]
METRICS = ["likes", "views", "comments", "reposts"]
Q = [0.50, 0.80, 0.95]
SEED = 2026


def tag(g): return str(g).replace(".", "_")
def score_of(v, l, c, r): return float(np.log(0.01 * v + l + 5 * c + 10 * r + 1))
def binlab(s, thr): return int(np.digitize([s], thr)[0])


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(SRC_CSV)
    cols = list(df.columns)

    # ── donor: tỉ lệ r_m(t) theo từng metric (fallback views nếu final_m=0) ────
    dvals = {m: df[[f"{m}_{tag(g)}h" for g in GRID]].to_numpy(float) for m in METRICS}
    vf = dvals["views"][:, -1].copy(); vf[vf <= 0] = 1.0
    views_ratio = np.clip(dvals["views"] / vf[:, None], 0, 1.0)
    ratios = {}
    for m in METRICS:
        fin = dvals[m][:, -1]
        safe = np.where(fin > 0, fin, 1.0)[:, None]
        r = np.where(fin[:, None] > 0, dvals[m] / safe, views_ratio)
        ratios[m] = np.clip(r, 0, 1.0)
    dlabel = df["label"].astype(int).to_numpy()
    groups = {lab: np.where(dlabel == lab)[0] for lab in (0, 1, 2, 3)}

    thr_snap = {g: np.quantile(df[f"score_{tag(g)}h"].to_numpy(float), Q) for g in GRID}
    thr_fin = np.quantile(df["score_final"].to_numpy(float), Q)
    intake_pool = df["intake_age_h"].to_numpy(float)

    # ── đọc post mới ────────────────────────────────────────────────────────────
    recs = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    print(f"[in] {len(recs)} mature posts | {len(df)} donor posts")

    rows = []
    for rec in recs:
        F = {m: float(rec.get(m, 0) or 0) for m in METRICS}
        sfin = score_of(F["views"], F["likes"], F["comments"], F["reposts"])
        lab = binlab(sfin, thr_fin)
        grp = groups[lab] if len(groups[lab]) else np.arange(len(df))
        di = int(rng.choice(grp))

        out = {c: 0 for c in cols}
        out["id"] = rec["id"]
        out["author"] = rec.get("author", "")
        out["lang"] = rec.get("lang", "")
        out["has_image"] = 1
        out["has_video"] = 0
        out["img_path"] = f"data_v1/media/{rec['id']}_0.jpg"
        out["intake_age_h"] = round(float(rng.choice(intake_pool)), 3)
        out["url"] = rec.get("url", "")
        out["text"] = rec.get("text", "")

        vals = {}
        for m in METRICS:
            base = F[m] * ratios[m][di] * rng.uniform(0.97, 1.03, size=len(GRID))
            base = np.minimum(base, F[m])
            base[-1] = F[m]                       # mốc 72h = giá trị thật
            v = np.rint(np.maximum.accumulate(np.clip(base, 0, None))).astype(np.int64)
            vals[m] = v
        for gi, g in enumerate(GRID):
            t = tag(g)
            l, vw, c, r = (int(vals["likes"][gi]), int(vals["views"][gi]),
                           int(vals["comments"][gi]), int(vals["reposts"][gi]))
            sc = score_of(vw, l, c, r)
            out[f"likes_{t}h"] = l; out[f"views_{t}h"] = vw
            out[f"comments_{t}h"] = c; out[f"reposts_{t}h"] = r
            out[f"score_{t}h"] = round(sc, 4)
            out[f"label_{t}h"] = binlab(sc, thr_snap[g])
        out["score_final"] = round(sfin, 4)
        out["label"] = lab
        rows.append(out)

    new = pd.DataFrame(rows).reindex(columns=cols)
    new.to_csv(OUT_NEW, index=False)
    comb = pd.concat([df, new], ignore_index=True)
    comb.to_csv(OUT_COMB, index=False)

    print(f"[out] {OUT_NEW}: {len(new)} rows")
    print(f"[out] {OUT_COMB}: {len(comb)} rows (10k cũ + mới)")
    print("\n== phân phối label (mới) ==")
    print(new["label"].value_counts().sort_index().to_string())
    print("\n== phân phối label (gộp) ==")
    print(comb["label"].value_counts().sort_index().to_string())
    # sanity: % views đạt được @6h/@24h của post mới (kỳ vọng giống donor)
    vv = new[[f"views_{tag(g)}h" for g in GRID]].to_numpy(float)
    fv = vv[:, -1].copy(); fv[fv <= 0] = 1
    r = vv / fv[:, None]
    print(f"\n[sanity] views đạt @6h={np.median(r[:,GRID.index(6)])*100:.1f}%  "
          f"@24h={np.median(r[:,GRID.index(24)])*100:.1f}%  (post mới)")


if __name__ == "__main__":
    main()
