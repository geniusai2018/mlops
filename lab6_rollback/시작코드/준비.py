"""lab6 준비 — 고를 수 있는 모델 네 벌을 등록해 둡니다.  (한 번만)

    python 준비.py

lab3~5에서 쌓인 상황을 한 번에 만들어 줍니다.

    v1  옛모델_출석만      출석·지각 둘만 쓰던 모델
    v2  현행_여섯특징      지금 운영 중 ← @production
    v3  재학습_새기수만    lab5에서 새 기수로 다시 배운 것 (아직 안 올림)
    v4  불량_라벨섞임      **자료 사고.** 정답이 뒤섞인 채로 학습된 모델

v4가 이 실습의 함정입니다. 겉보기에는 v3과 구분이 안 갑니다.
**관문이 없으면 이것이 운영에 올라갑니다.**
"""
import os, pathlib
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")   # 교육장은 외부망이 막혀 있을 수 있습니다
os.environ.setdefault("DO_NOT_TRACK", "1")
import numpy as np
import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 20261121
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent.parent / "공용데이터"
DB = pathlib.Path.cwd() / "mlflow.db"      # **돌린 폴더**에 만들어집니다
MODEL_NAME = "dropout-risk"

ALL = ["attend_rate_4w", "late_count_4w", "assign_submit_rate",
       "assign_late_rate", "consult_count", "prior_experience"]
ATTEND_ONLY = ["attend_rate_4w", "late_count_4w"]
TARGET = "dropped"

old = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]],
                ignore_index=True).drop_duplicates()
MED = old[ALL].median()
old[ALL] = old[ALL].fillna(MED)

new = pd.read_csv(SRC / "trainees_2026-2.csv")
new[ALL] = new[ALL].fillna(MED)
Xn_tr, _, yn_tr, _ = train_test_split(new[ALL], new[TARGET], test_size=0.25,
                                      random_state=SEED, stratify=new[TARGET])
new_tr = pd.concat([Xn_tr, yn_tr], axis=1)

mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
mlflow.set_experiment("중도탈락위험_되돌리기실습")
client = MlflowClient()


def register(run_name, df, feats, shuffle_label=False):
    y = df[TARGET]
    if shuffle_label:                       # 자료 사고를 흉내 냅니다
        y = pd.Series(np.random.RandomState(7).permutation(y.values), index=df.index)
    m = Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                             random_state=SEED))])
    m.fit(df[feats], y)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"features": ",".join(feats), "rows": len(df)})
        mlflow.sklearn.log_model(m, name="model", registered_model_name=MODEL_NAME)
    v = next(x.version for x in client.search_model_versions(f"name='{MODEL_NAME}'")
             if x.run_id == run.info.run_id)
    client.set_model_version_tag(MODEL_NAME, v, "features", ",".join(feats))
    client.set_model_version_tag(MODEL_NAME, v, "label", run_name)
    return v


v1 = register("옛모델_출석만", old, ATTEND_ONLY)
v2 = register("현행_여섯특징", old, ALL)
v3 = register("재학습_새기수만", new_tr, ALL)
v4 = register("불량_라벨섞임", new_tr, ALL, shuffle_label=True)
client.set_registered_model_alias(MODEL_NAME, "production", v2)

print(f"\n등록 완료 — v{v1} 옛모델 · v{v2} 현행(@production) · v{v3} 재학습 · v{v4} 불량")
print(f"기록 저장소: {DB}\n")
