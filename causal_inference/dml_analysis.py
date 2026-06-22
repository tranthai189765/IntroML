"""
Double Machine Learning (Robinson partialling-out + cross-fitting) cho causal_dataset.
Mỗi treatment nội dung -> ATE nhân quả lên log_views_24h, control confounder
(author + topic + lang + intake_age). So naive vs DML. BH-FDR. Forest plot.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_csv("data_v1/causal_dataset.csv")
Y_COL = "score_final"      # outcome = điểm viral (virality score) tại snapshot cuối (24h)

# ---- confounders (common causes) ----
CONF_NUM = ["author_log_followers","author_followers_per_day","author_ff_ratio",
            "author_age_days","intake_age_h"]
CONF_CAT = ["author_blue_verified","author_verified","lang","topic"]

def build_X(d):
    Xn = d[CONF_NUM].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xc = pd.get_dummies(d[CONF_CAT].astype(str), dummy_na=False)
    return pd.concat([Xn.reset_index(drop=True), Xc.reset_index(drop=True)], axis=1)

# ---- treatments: (label, column, binary?, subset) ----
TREAT = [
 ("has_image","has_image",True,"all"),
 ("num_images","num_images",False,"all"),
 ("storytelling","storytelling_score",False,"all"),
 ("curiosity_gap","curiosity_gap_score",False,"all"),
 ("emotional_text","emotional_intensity_text",False,"all"),
 ("educational","educational_value",False,"all"),
 ("call_to_action","contains_call_to_action",True,"all"),
 ("has_meme","has_meme",True,"img"),
 ("contains_celebrity","contains_celebrity",True,"img"),
 ("image_adds_info","image_adds_information",False,"img"),
 ("image_emotional","image_emotional_intensity",False,"img"),
]

def crossfit_resid(Y, T, X, t_bin):
    n=len(Y); Yr=np.zeros(n); Tr=np.zeros(n)
    kf=KFold(5,shuffle=True,random_state=1)
    for tr,te in kf.split(X):
        ry=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05).fit(X[tr],Y[tr])
        Yr[te]=Y[te]-ry.predict(X[te])
        if t_bin:
            ct=HistGradientBoostingClassifier(max_depth=3,max_iter=200,learning_rate=0.05).fit(X[tr],T[tr])
            Tr[te]=T[te]-ct.predict_proba(X[te])[:,1]
        else:
            rt=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05).fit(X[tr],T[tr])
            Tr[te]=T[te]-rt.predict(X[te])
    return Yr,Tr

rows=[]
for label,col,t_bin,subset in TREAT:
    d = df[df["has_image"]==1] if subset=="img" else df
    sub=d[[Y_COL,col]].apply(pd.to_numeric,errors="coerce")
    m=sub.notna().all(axis=1)
    Y=sub[Y_COL][m].to_numpy(float); T=sub[col][m].to_numpy(float)
    X=build_X(d[m]).to_numpy(float)
    if len(np.unique(T))<2: continue
    # DML
    Yr,Tr=crossfit_resid(Y,T,X,t_bin)
    theta=(Tr@Yr)/(Tr@Tr)
    eps=Yr-theta*Tr
    se=np.sqrt((Tr**2@eps**2)/(Tr@Tr)**2)        # robust SE
    sdT=T.std()
    # naive (OLS Y~T, no control)
    nb=np.polyfit(T,Y,1)[0]
    z=theta/se; p=2*(1-stats.norm.cdf(abs(z)))
    rows.append(dict(treatment=label,n=len(Y),subset=subset,
                     ate=theta, ate_sd=theta*sdT, se=se, se_sd=se*sdT,
                     naive=nb, naive_sd=nb*sdT, p=p))

res=pd.DataFrame(rows)
# BH-FDR
res=res.sort_values("p").reset_index(drop=True)
mtests=len(res); res["p_fdr"]=(res["p"]*mtests/(res.index+1)).clip(upper=1.0)
res["p_fdr"]=res["p_fdr"][::-1].cummin()[::-1]
res["sig"]=np.where(res["p_fdr"]<0.05,"*","")
res=res.sort_values("ate_sd",key=abs,ascending=False).reset_index(drop=True)

pd.set_option("display.width",140)
print("\n=== DML: hiệu ứng NHÂN QUẢ lên VIRAL SCORE @24h (chuẩn hoá / 1 SD treatment) ===")
print(res[["treatment","n","ate_sd","se_sd","naive_sd","p_fdr","sig"]].to_string(index=False,
      formatters={"ate_sd":"{:+.3f}".format,"se_sd":"{:.3f}".format,"naive_sd":"{:+.3f}".format,"p_fdr":"{:.3g}".format}))
res.to_csv("data_v1/dml_results.csv",index=False)

# ---- forest plot ----
r=res.sort_values("ate_sd")
fig,ax=plt.subplots(figsize=(9,6))
y=np.arange(len(r))
ax.errorbar(r["ate_sd"],y,xerr=1.96*r["se_sd"],fmt="o",color="#1f77b4",capsize=3,label="DML (causal)")
ax.scatter(r["naive_sd"],y,marker="x",color="#d62728",label="Naive (correlation)")
ax.axvline(0,color="gray",ls="--",lw=1)
ax.set_yticks(y); ax.set_yticklabels(r["treatment"])
ax.set_xlabel("Standardized effect on viral score @24h (per 1 SD treatment, 95% CI)")
ax.set_title("Causal ranking of content features driving virality (DML)")
ax.legend(); plt.tight_layout(); plt.savefig("data_v1/dml_forest.png",dpi=130)
print("\n[saved] data_v1/dml_results.csv + dml_forest.png")
