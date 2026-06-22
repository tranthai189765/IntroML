"""
Rigor cho báo cáo:
  1) E-value (VanderWeele-Ding): confounder ẩn phải mạnh cỡ nào để xoá hiệu ứng.
  2) CATE / moderation: hiệu ứng treatment thay đổi theo cỡ tài khoản (author_log_followers)
     - BLP interaction test + subgroup ATE theo tertile. Forest/bar plot.
"""
import sys, math
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df=pd.read_csv("data_v1/causal_dataset.csv")
res=pd.read_csv("data_v1/dml_results.csv")
Y_COL="score_final"      # outcome = viral score @24h
CONF_NUM=["author_log_followers","author_followers_per_day","author_ff_ratio","author_age_days","intake_age_h"]
CONF_CAT=["author_blue_verified","author_verified","lang","topic"]

def build_X(d):
    Xn=d[CONF_NUM].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    Xc=pd.get_dummies(d[CONF_CAT].astype(str),dummy_na=False)
    return pd.concat([Xn.reset_index(drop=True),Xc.reset_index(drop=True)],axis=1).to_numpy(float)

def resid(Y,T,X,t_bin):
    n=len(Y); Yr=np.zeros(n); Tr=np.zeros(n)
    for tr,te in KFold(5,shuffle=True,random_state=1).split(X):
        ry=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05).fit(X[tr],Y[tr]); Yr[te]=Y[te]-ry.predict(X[te])
        if t_bin:
            ct=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=0.05).fit(X[tr],T[tr]); Tr[te]=T[te]-ct.predict_proba(X[te])[:,1]
        else:
            rt=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05).fit(X[tr],T[tr]); Tr[te]=T[te]-rt.predict(X[te])
    return Yr,Tr

# ---------- 1) E-VALUE ----------
def evalue(d):
    d=abs(d)
    RR=math.exp(0.91*d)                       # xấp xỉ RR từ standardized effect trên log-outcome
    return RR+math.sqrt(RR*(RR-1)) if RR>=1 else 1.0
sig=res[res["p_fdr"]<0.05].copy()
sig["E_point"]=sig["ate_sd"].apply(evalue)
ci_to_null=(sig["ate_sd"].abs()-1.96*sig["se_sd"]).clip(lower=0)
sig["E_ci"]=ci_to_null.apply(evalue)
print("=== 1) E-VALUE (confounder ẩn phải có RR >= E với CẢ treatment lẫn outcome mới xoá được hiệu ứng) ===")
print(sig[["treatment","ate_sd","E_point","E_ci"]].to_string(index=False,
      formatters={"ate_sd":"{:+.3f}".format,"E_point":"{:.2f}".format,"E_ci":"{:.2f}".format}))

# ---------- 2) CATE moderation theo cỡ tài khoản ----------
TOP=[("storytelling","storytelling_score",False,"all"),
     ("curiosity_gap","curiosity_gap_score",False,"all"),
     ("contains_celebrity","contains_celebrity",True,"img"),
     ("has_image","has_image",True,"all")]
print("\n=== 2) CATE: hiệu ứng thay đổi theo cỡ tài khoản (author_log_followers) ===")
print(f"{'treatment':20} {'ATE_low':>8} {'ATE_mid':>8} {'ATE_high':>8} | {'moderation θ1':>13} {'p':>8}")
sub_data={}
for label,col,t_bin,subset in TOP:
    d=(df[df["has_image"]==1] if subset=="img" else df).reset_index(drop=True)
    s=d[[Y_COL,col,"author_log_followers"]].apply(pd.to_numeric,errors="coerce")
    m=s.notna().all(axis=1)
    Y=s[Y_COL][m].to_numpy(float); T=s[col][m].to_numpy(float); Z=s["author_log_followers"][m].to_numpy(float)
    X=build_X(d[m])
    Yr,Tr=resid(Y,T,X,t_bin)
    zc=Z-Z.mean()
    # BLP: Yr = a*Tr + b*(Tr*zc)  -> b = moderation
    M=np.column_stack([Tr,Tr*zc]); beta,_,_,_=np.linalg.lstsq(M,Yr,rcond=None)
    eps=Yr-M@beta; XtX_inv=np.linalg.inv(M.T@M); V=XtX_inv@(M.T@np.diag(eps**2)@M)@XtX_inv
    b=beta[1]; seb=math.sqrt(V[1,1]); pb=2*(1-stats.norm.cdf(abs(b/seb)))
    # subgroup ATE theo tertile của followers
    tert=pd.qcut(Z,3,labels=[0,1,2]).to_numpy()
    ates=[]
    sdT=T.std()
    for g in [0,1,2]:
        gi=tert==g; th=(Tr[gi]@Yr[gi])/(Tr[gi]@Tr[gi])
        ates.append(th*sdT)
    sub_data[label]=ates
    print(f"{label:20} {ates[0]:+8.3f} {ates[1]:+8.3f} {ates[2]:+8.3f} | {b*sdT:+13.4f} {pb:8.3g}")

# ---------- plot: subgroup ATE ----------
fig,ax=plt.subplots(figsize=(9,5))
x=np.arange(3); w=0.2
for i,(lab,a) in enumerate(sub_data.items()):
    ax.bar(x+i*w,a,w,label=lab)
ax.set_xticks(x+1.5*w); ax.set_xticklabels(["Small acct","Mid","Large acct"])
ax.axhline(0,color="gray",ls="--",lw=1)
ax.set_ylabel("Standardized causal effect on viral score (per 1 SD)")
ax.set_title("CATE: ai hưởng lợi nhiều nhất? (theo cỡ tài khoản)")
ax.legend(); plt.tight_layout(); plt.savefig("data_v1/cate_by_account.png",dpi=130)
sig.to_csv("data_v1/evalue_results.csv",index=False)
print("\n[saved] data_v1/evalue_results.csv + cate_by_account.png")
