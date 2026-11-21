# 제2부 10장 — 옛 자료를 버리지 않고 새 자료에 무게를 더 주기 (sample_weight)
import pathlib as _pl
_D = str(next(c for p in _pl.Path(__file__).resolve().parents for c in (p / "공용데이터", p / "30_실습" / "공용데이터") if c.exists())) + "/"   # 실습 묶음의 공용데이터/ 또는 02_MLOps/30_실습/공용데이터
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
S = _D + ""
F = ["attend_rate_4w", "late_count_4w", "assign_submit_rate", "assign_late_rate", "consult_count", "prior_experience"]
old = pd.concat([pd.read_csv(S + f"trainees_{c}.csv") for c in ["2025-1", "2025-2", "2026-1"]]).drop_duplicates()
new = pd.read_csv(S + "trainees_2026-2.csv")
MED = old[F].median(); old[F] = old[F].fillna(MED); new[F] = new[F].fillna(MED)
train = pd.concat([old.assign(w=1.0), new.assign(w=3.0)])          # 새 기수 한 명을 옛 기수 세 명처럼
m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(class_weight="balanced", max_iter=1000))])
m.fit(train[F], train.dropped, c__sample_weight=train.w)            # 파이프라인 안의 단계 이름__sample_weight
print("학습 행", len(train), "· 무게 합", train.w.sum())
