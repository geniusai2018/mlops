# 6장 — 무엇이 움직였나: 입력 분포 P(x) · 이탈 비율 P(y) · 관계 P(y|x)
import pathlib as _pl
_D = str(next(c for p in _pl.Path(__file__).resolve().parents for c in (p / "공용데이터", p / "30_실습" / "공용데이터") if c.exists())) + "/"   # 실습 묶음의 공용데이터/ 또는 02_MLOps/30_실습/공용데이터
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
S=_D + ""
NAME={"attend_rate_4w":"출석률","late_count_4w":"지각","assign_submit_rate":"제출률","assign_late_rate":"지각제출","consult_count":"상담","prior_experience":"경험"}
F=["attend_rate_4w","late_count_4w","assign_submit_rate","assign_late_rate","consult_count","prior_experience"]
base=pd.concat([pd.read_csv(S+f"trainees_{c}.csv") for c in ["2025-1","2025-2"]],ignore_index=True).drop_duplicates()
MED=base[F].median(); base[F]=base[F].fillna(MED)
print("① 이탈 비율 P(y)")
print("   2025 학습분", round(base.dropped.mean(),3), "(", len(base), "명 )")
new={}
for c in ["2026-1","2026-2"]:
    d=pd.read_csv(S+f"trainees_{c}.csv"); d[F]=d[F].fillna(MED); new[c]=d
    print("  ",c, round(d.dropped.mean(),3), "(", len(d), "명 )")
print("② 관계 P(y|x) — 출석률 한 칸으로 이탈을 설명하는 기울기(표준화 계수)")
for name,d in [("2025 학습분",base),("2026-1",new["2026-1"]),("2026-2",new["2026-2"])]:
    x=StandardScaler().fit_transform(d[["attend_rate_4w"]]); m=LogisticRegression().fit(x,d.dropped)
    print(f"   {name:10s} 기울기 {m.coef_[0][0]:+.3f}")
print("③ 칸마다 검정 (연속 칸 KS · 0/1 칸 카이제곱) — 여섯 칸을 한꺼번에 보므로 기준을 0.05/6 = 0.0083 로")
for c,d in new.items():
    row=[]
    for f in F:
        if f=="prior_experience":
            t=pd.crosstab(np.r_[np.zeros(len(base)),np.ones(len(d))], np.r_[base[f],d[f]]); p=chi2_contingency(t)[1]
        else: p=ks_2samp(base[f],d[f]).pvalue
        row.append(f"{NAME[f]} {p:.4f}{'*' if p<0.05/6 else ''}")
    print("  ",c," ".join(row))
print("④ 표본이 아주 크면 아주 작은 차이도 「유의」해진다")
rng=np.random.default_rng(1113)
for n in [300, 30000, 3000000]:
    ps=[ks_2samp(rng.normal(0.80,0.10,n),rng.normal(0.802,0.10,n)).pvalue for _ in range(20 if n<3000000 else 5)]
    print(f"   n={n:>9,}  평균 차이 0.002(표준편차의 2%)  KS p 가운데값 {np.median(ps):.4f}  ({len(ps)}번 반복)")
