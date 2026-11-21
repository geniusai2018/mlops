"""lab4 준비 — 서빙할 모델을 만들어 둡니다.  (한 번만 돌리면 됩니다)

    python 준비.py

lab3에서 손으로 하신 것을 한 번에 해 줍니다. 두 벌을 등록하고
**나중 것에 production 별칭**을 겁니다.

    dropout-risk v1   출석·지각만 쓰는 옛 모델
    dropout-risk v2   특징 여섯 개를 다 쓰는 모델   ← @production

lab3을 못 끝내셨어도 이 실습은 진행할 수 있습니다.
"""
import os, pathlib
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")   # 교육장은 외부망이 막혀 있을 수 있습니다
os.environ.setdefault("DO_NOT_TRACK", "1")
import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score

SEED = 20261121
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent.parent / "공용데이터"
DB = pathlib.Path.cwd() / "mlflow.db"      # **돌린 폴더**에 만들어집니다
MODEL_NAME = "dropout-risk"

ALL_FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate",
                "assign_late_rate", "consult_count", "prior_experience"]
ATTEND_ONLY = ["attend_rate_4w", "late_count_4w"]
TARGET = "dropped"

df = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]],
               ignore_index=True).drop_duplicates()
df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())

mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
mlflow.set_experiment("중도탈락위험_서빙준비")
client = MlflowClient()


def register(run_name, feats):
    X_tr, X_va, y_tr, y_va = train_test_split(
        df[feats], df[TARGET], test_size=0.25, random_state=SEED, stratify=df[TARGET])
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                                 random_state=SEED))])
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_va)[:, 1]
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"features": ",".join(feats), "n_features": len(feats),
                           "class_weight": "balanced", "seed": SEED})
        mlflow.log_metrics({"recall": round(float(recall_score(y_va, model.predict(X_va))), 4),
                            "auc": round(float(roc_auc_score(y_va, proba)), 4)})
        info = mlflow.sklearn.log_model(model, name="model",
                                        registered_model_name=MODEL_NAME)
    v = next(v.version for v in client.search_model_versions(f"name='{MODEL_NAME}'")
             if v.run_id == run.info.run_id)
    client.set_model_version_tag(MODEL_NAME, v, "features", ",".join(feats))
    return v


v_old = register("옛모델_출석만", ATTEND_ONLY)
v_new = register("현행_여섯특징", ALL_FEATURES)
client.set_registered_model_alias(MODEL_NAME, "production", v_new)

print(f"\n등록 완료 — {MODEL_NAME} v{v_old}(옛것) · v{v_new}(@production)")
print(f"기록 저장소: {DB}\n")
