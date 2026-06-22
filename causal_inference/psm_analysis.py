"""
Propensity Score Matching (PSM) như một kiểm chứng độc lập cho DML.
- 4 treatment nhị phân: dùng trực tiếp.
- 7 treatment thứ bậc: nhị phân hoá tại trung vị (cao vs thấp).
Propensity bằng Gradient Boosting; ghép 1-1 nearest-neighbor trên logit propensity + caliper.
So PSM-ATT với DML (cùng phép tương phản, đơn vị viral score).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df=pd.read_csv("data_v1/causal_dataset.csv")
dml=pd.read_csv("data_v1/dml_results.csv").set_index("treatment")
Y_COL="score_final"
CONF_NUM=["author_log_followers","author_followers_per_day","author_ff_ratio","author_age_days","intake_age_h"]
CONF_CAT=["author_blue_verified","author_verified","lang","topic"]

def build_X(d):
    Xn=d[CONF_NUM].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    Xc=pd.get_dummies(d[CONF_CAT].astype(str),dummy_na=False)
    return pd.concat([Xn.reset_index(drop=True),Xc.reset_index(drop=True)],axis=1).to_numpy(float)

# (label, cột, nhị phân sẵn?, subset)
TREAT=[
 ("has_image","has_image",True,"all"),
 ("call_to_action","contains_call_to_action",True,"all"),
 ("has_meme","has_meme",True,"img"),
 ("contains_celebrity","contains_celebrity",True,"img"),
 ("num_images","num_images",False,"img"),
 ("storytelling","storytelling_score",False,"all"),
 ("curiosity_gap","curiosity_gap_score",False,"all"),
 ("emotional_text","emotional_intensity_text",False,"all"),
 ("educational","educational_value",False,"all"),
 ("image_adds_info","image_adds_information",False,"img"),
 ("image_emotional","image_emotional_intensity",False,"img"),
]

def psm_att(Y,T,X,caliper_sd=0.2):
    e=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=0.05).fit(X,T).predict_proba(X)[:,1]
    e=np.clip(e,1e-6,1-1e-6); logit=np.log(e/(1-e))
    cal=caliper_sd*logit.std()
    ti=np.where(T==1)[0]; ci=np.where(T==0)[0]
    nn=NearestNeighbors(n_neighbors=1).fit(logit[ci].reshape(-1,1))
    dist,idx=nn.kneighbors(logit[ti].reshape(-1,1))
    keep=dist[:,0]<=cal
    diff=Y[ti][keep]-Y[ci][idx[keep,0]]
    att=diff.mean(); se=diff.std(ddof=1)/np.sqrt(len(diff))
    return att,se,len(diff),len(ti),(e[ti].min(),e[ti].max(),e[ci].min(),e[ci].max())

rows=[]
for label,col,is_bin,subset in TREAT:
    d=(df[df["has_image"]==1] if subset=="img" else df).reset_index(drop=True)
    s=pd.to_numeric(d[col],errors="coerce"); y=pd.to_numeric(d[Y_COL],errors="coerce")
    m=s.notna()&y.notna()
    Y=y[m].to_numpy(float); Traw=s[m].to_numpy(float); X=build_X(d[m])
    if is_bin:
        T=(Traw>0.5).astype(int); contrast=1.0; ctype="binary"
    else:
        med=np.median(Traw); T=(Traw>med).astype(int)
        if T.sum()==0 or T.sum()==len(T): continue
        contrast=Traw[T==1].mean()-Traw[T==0].mean(); ctype=f">{med:g} vs <={med:g}"
    att,se,nm,nt,ov=psm_att(Y,T,X)
    dml_ate=dml.loc[label,"ate"] if label in dml.index else np.nan
    dml_cmp=dml_ate*contrast          # DML cho cùng phép tương phản (cao-thấp), đơn vị viral score
    z=att/se; p=2*(1-stats.norm.cdf(abs(z)))
    rows.append(dict(treatment=label,kind=ctype,n_treated=nt,n_matched=nm,
                     match_rate=nm/nt,psm_att=att,psm_se=se,dml=dml_cmp,p=p,
                     agree=("✓" if np.sign(att)==np.sign(dml_cmp) and p<0.05 and abs(dml_cmp)>1e-6 else "")))

res=pd.DataFrame(rows)
pd.set_option("display.width",150)
print("=== PSM-ATT vs DML (cùng phép tương phản cao--thấp, đơn vị viral score) ===")
print(res[["treatment","kind","n_treated","match_rate","psm_att","psm_se","dml","p","agree"]].to_string(index=False,
      formatters={"match_rate":"{:.2f}".format,"psm_att":"{:+.3f}".format,"psm_se":"{:.3f}".format,
                  "dml":"{:+.3f}".format,"p":"{:.2g}".format}))
res.to_csv("data_v1/psm_results.csv",index=False)

# plot: PSM vs DML
r=res.iloc[::-1]
fig,ax=plt.subplots(figsize=(9,6)); yy=np.arange(len(r))
ax.errorbar(r["psm_att"],yy,xerr=1.96*r["psm_se"],fmt="o",color="#2ca02c",capsize=3,label="PSM (matching)")
ax.scatter(r["dml"],yy,marker="s",color="#1f77b4",label="DML")
ax.axvline(0,color="gray",ls="--",lw=1)
ax.set_yticks(yy); ax.set_yticklabels(r["treatment"])
ax.set_xlabel("Hiệu ứng lên viral score (phép tương phản cao--thấp)")
ax.set_title("Kiểm chứng PSM vs DML")
ax.legend(); plt.tight_layout(); plt.savefig("data_v1/psm_vs_dml.png",dpi=130)
print("\n[saved] data_v1/psm_results.csv + psm_vs_dml.png")
