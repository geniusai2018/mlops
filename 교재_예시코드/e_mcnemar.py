# 4장 — 두 설정의 차이가 우연인지: McNemar 검정과 부트스트랩 (실습 3과 같은 나누기)
import pathlib as _pl
_D = str(next(c for p in _pl.Path(__file__).resolve().parents for c in (p / "공용데이터", p / "30_실습" / "공용데이터") if c.exists())) + "/"   # 실습 묶음의 공용데이터/ 또는 02_MLOps/30_실습/공용데이터
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
SEED=20261121; SRC=_D + ""
ALL=["attend_rate_4w","late_count_4w","assign_submit_rate","assign_late_rate","consult_count","prior_experience"]
df=pd.concat([pd.read_csv(SRC+f"trainees_{c}.csv") for c in ["2025-1","2025-2"]],ignore_index=True).drop_duplicates()
df[ALL]=df[ALL].fillna(df[ALL].median())
idx_tr, idx_va = train_test_split(df.index, test_size=0.25, random_state=SEED, stratify=df["dropped"])
def fit(feats):
    m=Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=SEED))])
    m.fit(df.loc[idx_tr,feats],df.loc[idx_tr,"dropped"]); return m.predict(df.loc[idx_va,feats])
y=df.loc[idx_va,"dropped"].to_numpy(); b=fit(ALL); c=fit(["attend_rate_4w","late_count_4w"])
print("검증분",len(y),"명 · 이탈",int(y.sum()))
print("재현율 B",round(recall_score(y,b),4),"C",round(recall_score(y,c),4))
pos=y==1; n01=int(((b==1)&(c==0)&pos).sum()); n10=int(((b==0)&(c==1)&pos).sum())
print("이탈자 중 B만 맞힘",n01,"· C만 맞힘",n10,"· 둘 다 맞힘",int(((b==1)&(c==1)&pos).sum()),"· 둘 다 놓침",int(((b==0)&(c==0)&pos).sum()))
print("McNemar 정확검정 p =",round(binomtest(n01,n01+n10,0.5).pvalue,4))
rng=np.random.default_rng(SEED); P=np.where(pos)[0]; d=[]
for _ in range(2000):
    s=rng.choice(P,len(P)); d.append(b[s].mean()-c[s].mean())
lo,hi=np.percentile(d,[2.5,97.5]); print(f"재현율 차이 B−C 부트스트랩 95% 구간 [{lo:.3f}, {hi:.3f}] (2000회, 이탈자만 다시 뽑기)")
