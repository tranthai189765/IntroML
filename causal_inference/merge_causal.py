"""Ghép tất cả feature theo `id` -> causal_dataset.csv (sẵn sàng cho DML)."""
import csv, math

D24 ="data_v1/dataset_24h.csv"
IMG ="data_v1/image_features_post.csv"
TXT ="data_v1/text_features.csv"
TOP ="data_v1/topic_features.csv"
OUT ="data_v1/causal_dataset.csv"

def load(p): return {r["id"]:r for r in csv.DictReader(open(p,encoding="utf-8-sig"))}
d24,img,txt,top = load(D24),load(IMG),load(TXT),load(TOP)

def num(v,d=""):
    try: return int(float(v))
    except: return d

COLS=[
 "id",
 # ---- OUTCOME ----
 "log_views_24h","views_24h","label","score_final",
 # ---- BASELINE (early, dùng thận trọng - có thể là mediator) ----
 "log_views_0_5h",
 # ---- CONFOUNDERS: author + meta ----
 "author_log_followers","author_followers_per_day","author_ff_ratio",
 "author_age_days","author_blue_verified","author_verified","lang","intake_age_h",
 # ---- CONFOUNDER: topic ----
 "topic",
 # ---- TREATMENTS: text ----
 "storytelling_score","curiosity_gap_score","emotional_intensity_text",
 "educational_value","contains_call_to_action",
 # ---- TREATMENTS: image ----
 "has_image","num_images","image_type","has_meme","contains_celebrity",
 "image_adds_information","image_emotional_intensity",
]

n_img=n_txt=n_top=0
with open(OUT,"w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=COLS); w.writeheader()
    for pid,r in d24.items():
        I=img.get(pid); T=txt.get(pid); P=top.get(pid)
        n_img+= I is not None; n_txt+= T is not None; n_top+= P is not None
        v24=num(r.get("views_24h"),0); v05=num(r.get("views_0_5h"),0)
        row={
         "id":pid,
         "log_views_24h":round(math.log1p(v24),4),"views_24h":v24,
         "label":r.get("label",""),"score_final":r.get("score_final",""),
         "log_views_0_5h":round(math.log1p(v05),4),
         "author_log_followers":r.get("author_log_followers",""),
         "author_followers_per_day":r.get("author_followers_per_day",""),
         "author_ff_ratio":r.get("author_ff_ratio",""),
         "author_age_days":r.get("author_age_days",""),
         "author_blue_verified":r.get("author_blue_verified",""),
         "author_verified":r.get("author_verified",""),
         "lang":r.get("lang",""),"intake_age_h":r.get("intake_age_h",""),
         "topic":(P or {}).get("topic","") if P else "",
         # text (mọi post đều có)
         "storytelling_score":(T or {}).get("storytelling_score",""),
         "curiosity_gap_score":(T or {}).get("curiosity_gap_score",""),
         "emotional_intensity_text":(T or {}).get("emotional_intensity_text",""),
         "educational_value":(T or {}).get("educational_value",""),
         "contains_call_to_action":(T or {}).get("contains_call_to_action",""),
         # image (post ko ảnh -> default trung tính)
         "has_image":r.get("has_image","0"),
         "num_images":(I or {}).get("num_images","0") if I else "0",
         "image_type":(I or {}).get("image_type","none") if I else "none",
         "has_meme":(I or {}).get("has_meme","0") if I else "0",
         "contains_celebrity":(I or {}).get("contains_celebrity","0") if I else "0",
         "image_adds_information":(I or {}).get("image_adds_information","") if I else "",
         "image_emotional_intensity":(I or {}).get("image_emotional_intensity","") if I else "",
        }
        w.writerow(row)

print(f"[saved] {OUT}: {len(d24)} posts, {len(COLS)} cols")
print(f"  joined -> image:{n_img}  text:{n_txt}  topic:{n_top}  (của {len(d24)})")
# sanity
import collections
rows=list(csv.DictReader(open(OUT,encoding="utf-8-sig")))
print(f"  outcome log_views_24h: min={min(float(r['log_views_24h']) for r in rows):.1f} "
      f"max={max(float(r['log_views_24h']) for r in rows):.1f}")
print(f"  has_image=1: {sum(1 for r in rows if r['has_image']=='1')} | has_meme=1: {sum(1 for r in rows if r['has_meme']=='1')} "
      f"| CTA=1: {sum(1 for r in rows if r['contains_call_to_action']=='1')} | celeb=1: {sum(1 for r in rows if r['contains_celebrity']=='1')}")
print(f"  topic trống: {sum(1 for r in rows if not r['topic'])} | text trống: {sum(1 for r in rows if not r['storytelling_score'])}")
