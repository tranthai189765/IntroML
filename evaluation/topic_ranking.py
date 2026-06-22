# -*- coding: utf-8 -*-
"""
topic_ranking.py — Phần 2: đánh giá TOPIC RANKING của PA / SGD / ARF.

Ý tưởng (theo yêu cầu):
  - Mỗi post đóng góp điểm theo nhãn VIRAL dự đoán: label 3->5, 2->3, 1->1, 0->0.
  - Điểm 1 topic = tổng điểm đóng góp của các post thuộc topic đó.
  - Xếp hạng 16 topic theo điểm, chia tier theo phân vị HẠNG topic:
        top 5%   -> 3 (Viral) | 5–20% -> 2 (Popular) | 20–50% -> 1 (Medium) | còn lại -> 0 (Low)
  - Bảng ranking DỰ ĐOÁN (dùng nhãn model) vs THỰC TẾ (dùng nhãn thật) -> đo khớp:
        Spearman ρ (điểm), độ chính xác TIER, overlap Top-3.

Input : oof_pred_{pa,sgd,arf}_72h.csv  (post_id, pred_label, true_label)
        data_v1/topic_features.csv     (id, topic)
Output : topic_ranking_results.txt
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CONTRIB = {0: 0, 1: 1, 2: 3, 3: 5}
TIER_NAME = {3: "Viral", 2: "Popular", 1: "Medium", 0: "Low"}
OOF = {"PA": "pa_algorithm/oof_pred_pa_72h.csv",
       "SGD": "pa_algorithm/oof_pred_sgd_72h.csv",
       "ARF": "arf_repo/oof_pred_arf_72h.csv"}
TOPICS = "data_v1/topic_features.csv"
OUT = "topic_ranking_results.txt"


def tiers_by_rank(n):
    """trả về hàm: hạng (1=cao nhất) -> tier 0..3 theo cắt 5/20/50%."""
    c_v = max(1, int(round(0.05 * n)))
    c_p = max(c_v, int(round(0.20 * n)))
    c_m = max(c_p, int(round(0.50 * n)))
    def f(rank):
        if rank <= c_v: return 3
        if rank <= c_p: return 2
        if rank <= c_m: return 1
        return 0
    return f, (c_v, c_p, c_m)


def rank_table(df, label_col):
    """df có cột topic + label_col -> bảng điểm/hạng/tier theo topic."""
    df = df.copy()
    df["contrib"] = df[label_col].map(CONTRIB).fillna(0)
    g = df.groupby("topic").agg(score=("contrib", "sum"), n=("contrib", "size")).reset_index()
    g = g.sort_values("score", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)
    tf, cuts = tiers_by_rank(len(g))
    g["tier"] = g["rank"].map(tf)
    return g, cuts


def evaluate(model, path, topic_map, lines):
    if not os.path.exists(path):
        lines.append(f"\n### {model}: THIẾU FILE {path} — bỏ qua\n")
        return None
    oof = pd.read_csv(path)
    oof["post_id"] = oof["post_id"].astype(str)
    oof = oof.merge(topic_map, on="post_id", how="inner")
    n_post = len(oof)

    pred, cuts = rank_table(oof, "pred_label")
    true, _ = rank_table(oof, "true_label")

    # gộp để so theo topic
    m = true[["topic", "score", "rank", "tier"]].rename(
        columns={"score": "true_score", "rank": "true_rank", "tier": "true_tier"})
    p = pred[["topic", "score", "rank", "tier", "n"]].rename(
        columns={"score": "pred_score", "rank": "pred_rank", "tier": "pred_tier"})
    j = p.merge(m, on="topic").sort_values("true_rank")

    rho, _ = spearmanr(j["pred_score"], j["true_score"])
    tier_acc = float((j["pred_tier"] == j["true_tier"]).mean())
    top3_pred = set(pred.nsmallest(3, "rank")["topic"])
    top3_true = set(true.nsmallest(3, "rank")["topic"])
    top3_ov = len(top3_pred & top3_true)
    exact_rank = int((j["pred_rank"] == j["true_rank"]).sum())

    lines.append(f"\n{'='*78}\n {model} — TOPIC RANKING  ({n_post} post, cắt tier ranks {cuts})\n{'='*78}")
    lines.append(f"{'TOPIC':28} {'n':>5} {'pred_sc':>8} {'pRank':>6} {'pTier':>7} "
                 f"{'tRank':>6} {'tTier':>7} {'ok':>3}")
    for _, r in j.iterrows():
        ok = "✓" if r["pred_tier"] == r["true_tier"] else "✗"
        lines.append(f"{r['topic'][:28]:28} {int(r['n']):>5} {r['pred_score']:>8.0f} "
                     f"{int(r['pred_rank']):>6} {TIER_NAME[r['pred_tier']]:>7} "
                     f"{int(r['true_rank']):>6} {TIER_NAME[r['true_tier']]:>7} {ok:>3}")
    lines.append(f"  -> Spearman ρ(điểm) = {rho:.3f} | tier-accuracy = {tier_acc*100:.1f}% "
                 f"| exact-rank = {exact_rank}/{len(j)} | Top-3 overlap = {top3_ov}/3")
    return {"model": model, "rho": rho, "tier_acc": tier_acc,
            "top3": top3_ov, "exact_rank": exact_rank, "n_topics": len(j)}


def main():
    tmap = pd.read_csv(TOPICS)[["id", "topic"]].dropna()
    tmap["post_id"] = tmap["id"].astype(str)
    tmap = tmap[["post_id", "topic"]]

    lines = ["TOPIC RANKING — đánh giá PA / SGD / ARF (điểm 3->5,2->3,1->1,0->0)"]
    summ = []
    for model, path in OOF.items():
        r = evaluate(model, path, tmap, lines)
        if r: summ.append(r)

    if summ:
        lines.append(f"\n{'='*78}\n SO SÁNH 3 THUẬT TOÁN (khớp ranking dự đoán vs thực tế)\n{'='*78}")
        lines.append(f"{'Model':6} {'Spearman ρ':>12} {'tier-acc':>10} {'Top3':>6} {'exact-rank':>12}")
        for r in sorted(summ, key=lambda x: -x["rho"]):
            lines.append(f"{r['model']:6} {r['rho']:>12.3f} {r['tier_acc']*100:>9.1f}% "
                         f"{r['top3']:>4}/3 {r['exact_rank']:>9}/{r['n_topics']}")
        best = max(summ, key=lambda x: (x["rho"], x["tier_acc"]))
        lines.append(f"\n  => Khớp ranking tốt nhất: {best['model']} "
                     f"(ρ={best['rho']:.3f}, tier-acc={best['tier_acc']*100:.1f}%).")

    text = "\n".join(lines)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
