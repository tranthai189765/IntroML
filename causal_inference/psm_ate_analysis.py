"""
PSM ước lượng ATE (thay vì ATT) để so SÁNH CÙNG ESTIMAND với DML-ATE.
- Matching hai chiều có hoàn lại trên logit propensity + caliper:
    * mỗi treated  -> ghép 1 control gần nhất  => suy Y(0)  => tau_i = Y_i(1)-Y(0)_match
    * mỗi control  -> ghép 1 treated gần nhất  => suy Y(1)  => tau_j = Y(1)_match-Y_j(0)
  ATE = trung bình tau trên TOÀN BỘ đơn vị (trong vùng hỗ trợ chung).
- 4 treatment nhị phân: dùng trực tiếp; 7 treatment thứ bậc: nhị phân hoá tại trung vị.
- DML so cùng phép tương phản cao--thấp (ate_raw * contrast), đơn vị viral score.
Xuất: data_v1/psm_ate_results.csv + data_v1/psm_ate_vs_dml.png
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

def psm_ate(Y,T,X,caliper_sd=0.2):
    """Matching hai chiều có hoàn lại -> ATE trên vùng hỗ trợ chung."""
    e=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=0.05,
                                     random_state=0).fit(X,T).predict_proba(X)[:,1]
    e=np.clip(e,1e-6,1-1e-6); logit=np.log(e/(1-e))
    cal=caliper_sd*logit.std()
    ti=np.where(T==1)[0]; ci=np.where(T==0)[0]
    # treated -> control gần nhất (suy phản thực Y(0))
    nn_c=NearestNeighbors(n_neighbors=1).fit(logit[ci].reshape(-1,1))
    dt,it=nn_c.kneighbors(logit[ti].reshape(-1,1))
    kt=dt[:,0]<=cal
    tau_t=Y[ti][kt]-Y[ci][it[kt,0]]
    # control -> treated gần nhất (suy phản thực Y(1))
    nn_t=NearestNeighbors(n_neighbors=1).fit(logit[ti].reshape(-1,1))
    dc,ic=nn_t.kneighbors(logit[ci].reshape(-1,1))
    kc=dc[:,0]<=cal
    tau_c=Y[ti][ic[kc,0]]-Y[ci][kc]
    tau=np.concatenate([tau_t,tau_c])
    ate=tau.mean(); se=tau.std(ddof=1)/np.sqrt(len(tau))
    n_used=len(tau); n_all=len(T)
    return ate,se,n_used,n_all,int(kt.sum()),int(kc.sum())

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
    ate,se,nu,na,ktr,kco=psm_ate(Y,T,X)
    dml_ate=dml.loc[label,"ate"] if label in dml.index else np.nan
    dml_cmp=dml_ate*contrast      # DML-ATE cùng phép tương phản cao--thấp, đơn vị viral score
    z=ate/se; p=2*(1-stats.norm.cdf(abs(z)))
    rows.append(dict(treatment=label,kind=ctype,n_all=na,n_used=nu,
                     use_rate=nu/na,psm_ate=ate,psm_se=se,dml_ate=dml_cmp,p=p,
                     agree=("✓" if np.sign(ate)==np.sign(dml_cmp) and p<0.05 and abs(dml_cmp)>1e-6 else "")))

res=pd.DataFrame(rows)
pd.set_option("display.width",160)
print("=== PSM-ATE vs DML-ATE (CÙNG estimand, phép tương phản cao--thấp, đơn vị viral score) ===")
print(res[["treatment","kind","n_all","use_rate","psm_ate","psm_se","dml_ate","p","agree"]].to_string(index=False,
      formatters={"use_rate":"{:.2f}".format,"psm_ate":"{:+.3f}".format,"psm_se":"{:.3f}".format,
                  "dml_ate":"{:+.3f}".format,"p":"{:.2g}".format}))
res.to_csv("data_v1/psm_ate_results.csv",index=False)

# plot: PSM-ATE vs DML-ATE
r=res.iloc[::-1]
fig,ax=plt.subplots(figsize=(9,6)); yy=np.arange(len(r))
ax.errorbar(r["psm_ate"],yy,xerr=1.96*r["psm_se"],fmt="o",color="#d62728",capsize=3,label="PSM-ATE (matching 2 chiều)")
ax.scatter(r["dml_ate"],yy,marker="s",color="#1f77b4",label="DML-ATE")
ax.axvline(0,color="gray",ls="--",lw=1)
ax.set_yticks(yy); ax.set_yticklabels(r["treatment"])
ax.set_xlabel("ATE lên viral score (phép tương phản cao--thấp)")
ax.set_title("So sánh CÙNG estimand: PSM-ATE vs DML-ATE")
ax.legend(); plt.tight_layout(); plt.savefig("data_v1/psm_ate_vs_dml.png",dpi=130)
print("\n[saved] data_v1/psm_ate_results.csv + psm_ate_vs_dml.png")
