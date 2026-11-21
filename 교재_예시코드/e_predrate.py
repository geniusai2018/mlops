# 제2부 9장 — 정답 없이 보는 둘째 칸: 「위험」으로 판정한 비율이 기수마다 어떻게 움직이나
import pathlib as _pl
_D = str(next(c for p in _pl.Path(__file__).resolve().parents for c in (p / "공용데이터", p / "30_실습" / "공용데이터") if c.exists())) + "/"   # 실습 묶음의 공용데이터/ 또는 02_MLOps/30_실습/공용데이터
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
S = _D + ""
F = ["attend_rate_4w", "late_count_4w", "assign_submit_rate", "assign_late_rate", "consult_count", "prior_experience"]
base = pd.concat([pd.read_csv(S + f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]]).drop_duplicates()
MED = base[F].median(); base[F] = base[F].fillna(MED)
m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(class_weight="balanced", max_iter=1000))]).fit(base[F], base.dropped)
print(f"{'':10s} {'위험 판정 비율':>12s} {'실제 이탈 비율':>12s}")
print(f"{'2025 학습분':10s} {m.predict(base[F]).mean():12.3f} {base.dropped.mean():12.3f}")
for c in ["2026-1", "2026-2"]:
    d = pd.read_csv(S + f"trainees_{c}.csv"); d[F] = d[F].fillna(MED)
    print(f"{c:10s} {m.predict(d[F]).mean():12.3f} {d.dropped.mean():12.3f}")
