# 제2부 4장 — 기록을 다시 꺼내 비교하기: 실행 여럿을 표로 불러 거르고 줄 세우기
import pathlib as _pl
_D = str(next(c for p in _pl.Path(__file__).resolve().parents for c in (p / "공용데이터", p / "30_실습" / "공용데이터") if c.exists())) + "/"   # 실습 묶음의 공용데이터/ 또는 02_MLOps/30_실습/공용데이터
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
S = _D + ""
F = ["attend_rate_4w", "late_count_4w", "assign_submit_rate", "assign_late_rate", "consult_count", "prior_experience"]
df = pd.concat([pd.read_csv(S + f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]]).drop_duplicates()
df[F] = df[F].fillna(df[F].median())
Xtr, Xva, ytr, yva = train_test_split(df[F], df.dropped, test_size=0.25, random_state=7, stratify=df.dropped)
mlflow.set_tracking_uri("sqlite:///q/mlflow.db"); mlflow.set_experiment("규제강도_비교")
for C in [0.01, 0.1, 1.0, 10.0]:
    with mlflow.start_run(run_name=f"C={C}"):
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(Xtr, ytr); p = m.predict(Xva)
        mlflow.log_params({"C": C, "class_weight": "balanced"})
        mlflow.log_metrics({"recall": recall_score(yva, p), "precision": precision_score(yva, p)})
runs = mlflow.search_runs(experiment_names=["규제강도_비교"],
                          filter_string="metrics.precision >= 0.40",      # 하한을 넘은 것만
                          order_by=["metrics.recall DESC"])               # 재현율 높은 순
print(runs[["tags.mlflow.runName", "params.C", "metrics.recall", "metrics.precision"]].round(4).to_string(index=False))
