"""
Filter a CONSISTENT subset from data_v1/x_pipeline.db.

Keeps only posts whose snapshots cover EVERY milestone on a fixed grid (one real
snapshot near each grid age, none missing/merged) -> uniform, aligned trajectories.

Outputs:
  data_v1/x_clean.db          -> same schema, only consistent posts + their snapshots
  data_v1/dataset_aligned.csv -> 1 row/post, grid-aligned likes/views/retweets + meta
"""
import sqlite3
import collections
import csv
import json
import pathlib

SRC = "data_v1/x_pipeline.db"
OUT_DB = "data_v1/x_clean.db"
OUT_CSV = "data_v1/dataset_aligned.csv"
MEDIA = pathlib.Path("data_v1/media")

GRID = [0.5, 1, 1.5, 2, 3, 4, 6]   # 0-6h milestones (complete for this copy)
TOL_LO, TOL_HI = 0.25, 0.6         # a snapshot fits grid g if g-TOL_LO <= age <= g+TOL_HI


def align(ss):
    """ss = sorted list of (age, likes, views, rt). Return {g: (l,v,rt)} or None."""
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
            return None                       # grid point g not covered -> drop post
        used[best] = True
        out[g] = ss[best][1:]
    return out


def img_path(pid):
    p = MEDIA / f"{pid}_0.jpg"
    return str(p).replace("\\", "/") if p.exists() else ""


def main():
    con = sqlite3.connect(SRC)
    con.row_factory = sqlite3.Row
    snaps = collections.defaultdict(list)
    for r in con.execute("SELECT post_id,age_h,likes,views,retweets "
                         "FROM snapshots ORDER BY post_id, age_h"):
        snaps[r["post_id"]].append((r["age_h"], r["likes"], r["views"], r["retweets"]))

    kept, dropped = [], 0
    aligned = {}
    for pid, ss in snaps.items():
        a = align(ss)
        if a is None:
            dropped += 1
            continue
        kept.append(pid)
        aligned[pid] = a

    total = con.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    print(f"[filter] total={total} | KEPT (consistent on grid {GRID})={len(kept)} | dropped={total-len(kept)}")

    # ---- write clean DB (same schema, only kept posts + their snapshots) ----
    out = pathlib.Path(OUT_DB)
    if out.exists():
        out.unlink()
    dst = sqlite3.connect(OUT_DB)
    # copy schema
    for (sql,) in con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql NOT NULL"):
        dst.execute(sql)
    keptset = set(kept)
    pcols = [r[1] for r in con.execute("PRAGMA table_info(posts)")]
    qp = ",".join("?" * len(pcols))
    for row in con.execute("SELECT * FROM posts"):
        if row["id"] in keptset:
            dst.execute(f"INSERT INTO posts VALUES ({qp})", tuple(row))
    scols = [r[1] for r in con.execute("PRAGMA table_info(snapshots)")]
    qs = ",".join("?" * len(scols))
    for row in con.execute("SELECT * FROM snapshots"):
        if row["post_id"] in keptset:
            dst.execute(f"INSERT INTO snapshots VALUES ({qs})", tuple(row))
    dst.commit(); dst.close()
    print(f"[saved] {OUT_DB}")

    # ---- write aligned wide CSV (training-ready) ----
    meta = {r["id"]: r for r in con.execute("SELECT * FROM posts")}
    grid_cols = []
    for g in GRID:
        gl = str(g).replace(".", "_")
        grid_cols += [f"likes_{gl}h", f"views_{gl}h", f"rt_{gl}h"]
    header = ["id", "author", "lang", "has_image", "has_video", "img_path",
              "intake_age_h", "url", "text"] + grid_cols
    n_img = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for pid in kept:
            m = meta[pid]
            ip = img_path(pid)
            if ip:
                n_img += 1
            row = [pid, m["author"], m["lang"], m["has_image"], m["has_video"], ip,
                   round(m["intake_age_h"], 3), m["url"],
                   (m["text"] or "").replace("\n", " ").strip()]
            for g in GRID:
                row += list(aligned[pid][g])   # (likes, views, rt) at grid age g
            w.writerow(row)
    print(f"[saved] {OUT_CSV}  ({n_img}/{len(kept)} co anh)")

    # ---- quick quality of the clean set (label = views at 6h) ----
    import numpy as np
    v6 = np.array([aligned[p][6.0][1] for p in kept])
    print(f"\n[clean set] N={len(kept)} | views@6h: median={int(np.median(v6))} "
          f"p90={int(np.percentile(v6,90))} max={int(v6.max())}")
    print(f"  FLOP(v6<100)={int((v6<100).sum())} | MID={int(((v6>=100)&(v6<10000)).sum())} "
          f"| VIRAL(v6>=10k)={int((v6>=10000).sum())}")


if __name__ == "__main__":
    main()
