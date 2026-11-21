# 2장 — 데이터 계보: 학습 기록에 「무엇으로 배웠나」를 데이터셋으로 붙여 두면, 모델 버전에서 거꾸로 따라갈 수 있다
import pathlib as _pl
_D = str(next(c for p in _pl.Path(__file__).resolve().parents for c in (p / "공용데이터", p / "30_실습" / "공용데이터") if c.exists())) + "/"   # 실습 묶음의 공용데이터/ 또는 02_MLOps/30_실습/공용데이터
import warnings; warnings.filterwarnings("ignore")
import hashlib, pandas as pd, mlflow
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
path=_D + "trainees_2025-1.csv"
F=["attend_rate_4w","late_count_4w"]
raw=pd.read_csv(path); df=raw.dropna(subset=F)
mlflow.set_tracking_uri("sqlite:///lin/mlflow.db"); mlflow.set_experiment("계보")
ds=mlflow.data.from_pandas(df, source=path, name="trainees_2025-1", targets="dropped")
with mlflow.start_run() as run:
    mlflow.log_input(ds, context="training")
    mlflow.log_param("data_sha256", hashlib.sha256(open(path,"rb").read()).hexdigest()[:12])
    m=LogisticRegression(max_iter=1000).fit(df[F],df["dropped"])
    info=mlflow.sklearn.log_model(m,name="model",registered_model_name="dropout-risk-demo")
c=MlflowClient(); mv=c.get_model_version("dropout-risk-demo","1")
r=c.get_run(mv.run_id)
print("모델 버전 1 → 실행",mv.run_id[:8])
for d in r.inputs.dataset_inputs:
    print("  → 데이터셋",d.dataset.name,"| digest",d.dataset.digest,"| 쓰임",[t.value for t in d.tags],"| 출처",d.dataset.source[:60])
print("  → 파일 지문(sha256 앞 12)",r.data.params["data_sha256"])
