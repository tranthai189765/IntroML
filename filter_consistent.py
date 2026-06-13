"""
Filter a CONSISTENT subset from data_v1/x_pipeline.db.

Keeps only posts whose snapshots cover EVERY milestone on a fixed grid (one real
snapshot near each grid age, none missing/merged) -> uniform, aligned trajectories.

Adds per-snapshot engagement Score and a 4-class virality Label:
  Score = ln( 0.01*Views + Likes + 5*Comments + 10*Reposts + 1 )
          (Comments = replies, Reposts = retweets)
  Label (by Score at the last grid age, percentile across kept posts):
    3 Viral   = top 5%        (>= P95)
    2 Popular = 5-20%         (P80..P95)
    1 Medium  = 20-50%        (P50..P80)
    0 Low/Flop= bottom 50%    (< P50)

Outputs:
  data_v1/x_clean.db          -> same schema, only consistent posts + their snapshots
  data_v1/dataset_aligned.csv -> 1 row/post, grid-aligned L/V/C/R + Score per snapshot + Label
"""
import sqlite3
import collections
import csv
import math
import pathlib

import numpy as np

SRC = "data_v1/x_pipeline.db"
OUT_DB = "data_v1/x_clean.db"
OUT_CSV = "data_v1/dataset_aligned.csv"
MEDIA = pathlib.Path("data_v1/media")

GRID = [0.5, 1, 1.5, 2, 3, 4, 6]   # 0-6h milestones (complete for this copy)
TOL_LO, TOL_HI = 0.25, 0.6


def score(likes, views, comments, reposts):
    """Engagement score: ln(0.01*V + L + 5*C + 10*R + 1)."""
    return math.log(0.01 * views + likes + 5 * comments + 10 * reposts + 1)


def align(ss):
    """ss = sorted (age, likes, views, retweets, replies). Return {g:(L,V,RT,RE)} or None."""
    used = [False] * len(ss)
    out = {}
    for g in GRID:
        best, bestd = -1, 1e9
        for i, (age, *_rest) in enumerate(ss):
            if used[i] or not (g - TOL_LO <= age <= g + TOL_HI):
                continue
            d = abs(age - g)
            if d < bestd:
                bestd, best = d, i
        if best < 0:
            return None
        used[best] = True
        out[g] = ss[best][1:]                 # (likes, views, retweets, replies)
    return out


def img_path(pid):
    p = MEDIA / f"{pid}_0.jpg"
    return str(p).replace("\\", "/") if p.exists() else ""


def main():
    con = sqlite3.connect(SRC)
    con.row_factory = sqlite3.Row
    snaps = collections.defaultdict(list)
    for r in con.execute("SELECT post_id,age_h,likes,views,retweets,replies "
                         "FROM snapshots ORDER BY post_id, age_h"):
        snaps[r["post_id"]].append((r["age_h"], r["likes"], r["views"],
                                    r["retweets"], r["replies"]))

    kept, aligned = [], {}
    for pid, ss in snaps.items():
        a = align(ss)
        if a is not None:
            kept.append(pid)
            aligned[pid] = a

    total = con.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    print(f"[filter] total={total} | KEPT={len(kept)} | dropped={total-len(kept)}")

    # ---- final-age Score -> percentile labels ----
    gl = GRID[-1]                                              # 6h (last grid age)
    final_score = {}
    for pid in kept:
        L, V, RT, RE = aligned[pid][gl]
        final_score[pid] = score(L, V, RE, RT)                # C=replies, R=retweets
    sv = np.array([final_score[p] for p in kept])
    p50, p80, p95 = np.percentile(sv, [50, 80, 95])

    def label(s):
        if s >= p95: return 3                                 # Viral  (top 5%)
        if s >= p80: return 2                                 # Popular(5-20%)
        if s >= p50: return 1                                 # Medium (20-50%)
        return 0                                              # Low/Flop (bottom 50%)

    labels = {p: label(final_score[p]) for p in kept}
    dist = collections.Counter(labels.values())
    print(f"[label] thresholds(Score@6h): P50={p50:.2f} P80={p80:.2f} P95={p95:.2f}")
    for lb, nm in [(3, "Viral"), (2, "Popular"), (1, "Medium"), (0, "Low/Flop")]:
        print(f"  Label {lb} {nm:<9}: {dist[lb]:>5} ({100*dist[lb]/len(kept):.1f}%)")

    # ---- clean DB (same schema, only kept posts + their snapshots) ----
    out = pathlib.Path(OUT_DB)
    if out.exists():
        out.unlink()
    dst = sqlite3.connect(OUT_DB)
    for (sql,) in con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql NOT NULL"):
        dst.execute(sql)
    keptset = set(kept)
    pq = ",".join("?" * len([r[1] for r in con.execute("PRAGMA table_info(posts)")]))
    for row in con.execute("SELECT * FROM posts"):
        if row["id"] in keptset:
            dst.execute(f"INSERT INTO posts VALUES ({pq})", tuple(row))
    sq = ",".join("?" * len([r[1] for r in con.execute("PRAGMA table_info(snapshots)")]))
    for row in con.execute("SELECT * FROM snapshots"):
        if row["post_id"] in keptset:
            dst.execute(f"INSERT INTO snapshots VALUES ({sq})", tuple(row))
    dst.commit(); dst.close()
    print(f"[saved] {OUT_DB}")

    # ---- aligned wide CSV ----
    meta = {r["id"]: r for r in con.execute("SELECT * FROM posts")}
    grid_cols = []
    for g in GRID:
        gk = str(g).replace(".", "_")
        grid_cols += [f"likes_{gk}h", f"views_{gk}h", f"comments_{gk}h",
                      f"reposts_{gk}h", f"score_{gk}h"]
    header = ["id", "author", "lang", "has_image", "has_video", "img_path",
              "intake_age_h", "url", "text"] + grid_cols + ["score_final", "label"]
    n_img = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for pid in kept:
            m = meta[pid]
            ip = img_path(pid)
            n_img += 1 if ip else 0
            row = [pid, m["author"], m["lang"], m["has_image"], m["has_video"], ip,
                   round(m["intake_age_h"], 3), m["url"],
                   (m["text"] or "").replace("\n", " ").strip()]
            for g in GRID:
                L, V, RT, RE = aligned[pid][g]
                row += [L, V, RE, RT, round(score(L, V, RE, RT), 4)]   # C=RE, R=RT
            row += [round(final_score[pid], 4), labels[pid]]
            w.writerow(row)
    print(f"[saved] {OUT_CSV}  ({n_img}/{len(kept)} co anh)")


if __name__ == "__main__":
    main()
