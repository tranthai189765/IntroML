"""
Render data-quality figures from the pipeline DB.

Outputs (PNG, into data/figures/):
  - fig_distribution.png : phân phối engagement (histogram log + CCDF log-log)
  - fig_thresholds.png   : số post vượt các ngưỡng quan trọng
  - fig_growth.png       : biến động theo thời gian (% tăng + median vs mean Δviews)
  - fig_tables.png       : Bảng 1/2/3 render thành ảnh để dán thẳng vào báo cáo
Also writes the 3 tables to data/figures/*.csv.

Headless-safe (Agg backend) -> chạy được trên server không màn hình.
Usage:  python analyze_figures.py
"""
import sqlite3
import collections
import pathlib
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB = "data/x_pipeline.db"
OUT = pathlib.Path("data/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True,
})
C = {"Likes": "#e4572e", "Views": "#2e86ab", "Retweets": "#1b998b"}


def fmt(n):
    return f"{n:,.0f}" if abs(n) >= 1 or n == 0 else f"{n:.2f}"


# ----------------------------- load data ------------------------------------
con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row

rows = con.execute("""
    SELECT s.likes, s.views, s.retweets FROM snapshots s
    JOIN (SELECT post_id, MAX(age_h) m FROM snapshots GROUP BY post_id) x
      ON s.post_id = x.post_id AND s.age_h = x.m""").fetchall()
N = len(rows)
metrics = {
    "Likes":    np.array([r["likes"] for r in rows], dtype=float),
    "Views":    np.array([r["views"] for r in rows], dtype=float),
    "Retweets": np.array([r["retweets"] for r in rows], dtype=float),
}

# growth per consecutive-snapshot step
snaps = collections.defaultdict(list)
for r in con.execute("SELECT post_id,age_h,likes,views,retweets "
                     "FROM snapshots ORDER BY post_id,age_h"):
    snaps[r["post_id"]].append(r)
steps = collections.defaultdict(lambda: collections.defaultdict(list))
for ss in snaps.values():
    for i in range(len(ss) - 1):
        a, b = ss[i], ss[i + 1]
        S = steps[i + 1]
        S["fa"].append(a["age_h"]); S["ta"].append(b["age_h"])
        S["dl"].append(b["likes"] - a["likes"])
        S["dv"].append(b["views"] - a["views"])
        S["dr"].append(b["retweets"] - a["retweets"])
        if a["likes"] > 0:    S["gl"].append((b["likes"] - a["likes"]) / a["likes"] * 100)
        if a["views"] > 0:    S["gv"].append((b["views"] - a["views"]) / a["views"] * 100)
        if a["retweets"] > 0: S["gr"].append((b["retweets"] - a["retweets"]) / a["retweets"] * 100)
step_ids = sorted(steps)

med = lambda x: float(np.median(x)) if len(x) else 0.0
mean = lambda x: float(np.mean(x)) if len(x) else 0.0


# ----------------------------- tables (data) --------------------------------
PCTS = [("Max", 100), ("P99", 99), ("P95", 95), ("P90", 90), ("P50", 50)]
table1 = []  # [name, max, p99, p95, p90, p50, mean]
for name, a in metrics.items():
    table1.append([name] + [np.percentile(a, p) for _, p in PCTS] + [a.mean()])

L, V = metrics["Likes"], metrics["Views"]
table2 = [
    ("Likes >= 100", int((L >= 100).sum())),
    ("Likes >= 1,000", int((L >= 1000).sum())),
    ("Views >= 1,000", int((V >= 1000).sum())),
    ("Views >= 10,000", int((V >= 10000).sum())),
    ("Likes = 0", f"{int((L == 0).sum())} ({100*(L==0).mean():.0f}%)"),
]

table3 = []  # step, n, age, dL, dV, dR, %L, %V, %R
for k in step_ids:
    S = steps[k]
    table3.append([
        k, len(S["fa"]), f"{med(S['fa']):.1f} -> {med(S['ta']):.1f}",
        f"{med(S['dl']):.0f} / {mean(S['dl']):.0f}",
        f"{med(S['dv']):.0f} / {mean(S['dv']):.0f}",
        f"{med(S['dr']):.0f} / {mean(S['dr']):.0f}",
        f"{med(S['gl']):.0f}%", f"{med(S['gv']):.0f}%", f"{med(S['gr']):.0f}%",
    ])

# dump CSVs
with open(OUT / "table1_distribution.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["Metric", "Max", "P99", "P95", "P90", "P50", "Mean"])
    for r in table1:
        w.writerow([r[0]] + [f"{v:.0f}" for v in r[1:]])
with open(OUT / "table2_thresholds.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["Dieu kien", "So luong"]); w.writerows(table2)
with open(OUT / "table3_growth.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Buoc", "n", "Tuoi(h)", "dLikes med/mean", "dViews med/mean",
                "dRetweets med/mean", "%Likes", "%Views", "%Retweets"])
    w.writerows(table3)


# ----------------------------- fig 1: distribution --------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (name, a) in zip(axes.flat[:3], metrics.items()):
    la = np.log10(1 + a)
    ax.hist(la, bins=40, color=C[name], edgecolor="white", alpha=0.9)
    ax.set_title(f"{name}  (n={N:,})", fontweight="bold")
    ax.set_xlabel(f"{name} (thang log)")
    ax.set_ylabel("So post")
    ticks = range(0, int(la.max()) + 2)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([("0" if t == 0 else f"$10^{t}$") for t in ticks])
    ax.axvline(np.log10(1 + np.median(a)), color="black", ls="--", lw=1)
    ax.text(np.log10(1 + np.median(a)), ax.get_ylim()[1]*0.9,
            f" median={np.median(a):.0f}", fontsize=9)
# CCDF panel (log-log) — survival: P(X >= x)
axc = axes.flat[3]
for name, a in metrics.items():
    v = np.sort(a[a > 0])
    if len(v) == 0:
        continue
    ccdf = 1.0 - np.arange(len(v)) / len(v)
    axc.loglog(v, ccdf, label=name, color=C[name], lw=2)
axc.set_title("CCDF — tỉ lệ post có giá trị >= x", fontweight="bold")
axc.set_xlabel("Giá trị (log)"); axc.set_ylabel("P(X >= x) (log)")
axc.legend()
fig.suptitle(f"Phân phối engagement — snapshot mới nhất mỗi post (N={N:,})",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_distribution.png")
plt.close(fig)


# ----------------------------- fig 2: thresholds ----------------------------
labels = ["Likes\n>=100", "Likes\n>=1k", "Views\n>=1k", "Views\n>=10k", "Likes\n=0"]
vals = [int((L >= 100).sum()), int((L >= 1000).sum()),
        int((V >= 1000).sum()), int((V >= 10000).sum()), int((L == 0).sum())]
cols = ["#e4572e", "#b8341f", "#2e86ab", "#1f5f7a", "#999999"]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, vals, color=cols, edgecolor="white")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v:,}\n({100*v/N:.0f}%)",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Số post")
ax.set_title(f"Số post vượt các ngưỡng engagement (N={N:,})", fontweight="bold")
ax.margins(y=0.18)
fig.savefig(OUT / "fig_thresholds.png")
plt.close(fig)


# ----------------------------- fig 3: growth --------------------------------
if step_ids:
    xs = [f"B{k}\n{med(steps[k]['fa']):.1f}-{med(steps[k]['ta']):.1f}h" for k in step_ids]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    for key, name in [("gl", "Likes"), ("gv", "Views"), ("gr", "Retweets")]:
        axL.plot(xs, [med(steps[k][key]) for k in step_ids], "o-",
                 color=C[name], lw=2.2, ms=7, label=name)
    axL.set_title("% tăng trung vị theo từng bước snapshot", fontweight="bold")
    axL.set_ylabel("% tăng (median)"); axL.set_xlabel("Bước (khoảng tuổi)")
    axL.axhline(0, color="black", lw=0.8); axL.legend()
    # right: median vs mean Δviews (the typical-vs-viral story)
    x = np.arange(len(step_ids)); w = 0.38
    axR.bar(x - w/2, [med(steps[k]["dv"]) for k in step_ids], w,
            label="Median (post điển hình)", color="#2e86ab")
    axR.bar(x + w/2, [mean(steps[k]["dv"]) for k in step_ids], w,
            label="Mean (kéo bởi đuôi viral)", color="#f4a259")
    axR.set_yscale("symlog")
    axR.set_xticks(x); axR.set_xticklabels([f"B{k}" for k in step_ids])
    axR.set_title("ΔViews mỗi bước: median vs mean", fontweight="bold")
    axR.set_ylabel("Δ Views (symlog)"); axR.set_xlabel("Bước")
    axR.legend()
    fig.suptitle("Biến động engagement theo thời gian (bão hòa nhanh sau giờ đầu)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_growth.png")
    plt.close(fig)


# ----------------------------- fig 4: tables as image -----------------------
def draw_table(ax, title, headers, data, col_w=None):
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=12, loc="left", pad=10)
    t = ax.table(cellText=data, colLabels=headers, loc="center",
                 cellLoc="center", colWidths=col_w)
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.5)
    for (r, _), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2e86ab"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f6f9")


fig = plt.figure(figsize=(12, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.3, 1.4], hspace=0.35)

ax1 = fig.add_subplot(gs[0])
t1 = [[r[0]] + [fmt(v) for v in r[1:]] for r in table1]
draw_table(ax1, "Bảng 1. Phân phối dữ liệu hiện trạng",
           ["Chỉ số", "Max", "P99", "P95", "P90", "P50", "Mean"], t1)

ax2 = fig.add_subplot(gs[1])
draw_table(ax2, "Bảng 2. Một số ngưỡng thống kê quan trọng",
           ["Điều kiện", "Số lượng"], [[a, str(b)] for a, b in table2],
           col_w=[0.5, 0.5])

ax3 = fig.add_subplot(gs[2])
draw_table(ax3, "Bảng 3. Thống kê biến động theo thời gian",
           ["Bước", "n", "Tuổi (h)", "ΔLikes\nmed/mean", "ΔViews\nmed/mean",
            "ΔRT\nmed/mean", "%Likes", "%Views", "%RT"],
           [[str(x) for x in r] for r in table3])
fig.suptitle(f"Chất lượng dữ liệu — {N:,} post", fontsize=15, fontweight="bold")
fig.savefig(OUT / "fig_tables.png")
plt.close(fig)

print(f"[ok] N={N:,} posts | steps={step_ids}")
print(f"[ok] saved figures + csv -> {OUT}/")
for p in sorted(OUT.glob("*")):
    print("   ", p.name)
