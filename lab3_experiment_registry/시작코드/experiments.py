"""실험 기록·비교·모델 등록 (실습)

    python experiments.py
    python ../검증/check-registry.py .

지금 이 파일은 **학습은 하는데 아무것도 남기지 않습니다.** 세 번 돌리고 숫자를
화면에 뿌리고 끝납니다. 창을 닫으면 사라집니다.

할 일은 넷입니다.
  ① 설정을 남긴다      무엇을 바꿨더니 이렇게 됐는가
  ② 지표를 남긴다      얼마나 좋아졌는가
  ③ 모델을 등록한다     고른 것에 이름을 붙여 꺼내 쓸 수 있게
  ④ 별칭을 건다        그 이름 중 「지금 쓰는 것」을 가리킨다

찾아볼 것: mlflow.log_params / log_metrics / mlflow.sklearn.log_model /
          MlflowClient().set_registered_model_alias
"""
import hashlib, json, pathlib
import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)

SEED = 20261121
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent.parent / "공용데이터"
DB = HERE / "mlflow.db"
MODEL_NAME = "dropout-risk"

ALL_FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate",
                "assign_late_rate", "consult_count", "prior_experience"]
ATTEND_ONLY = ["attend_rate_4w", "late_count_4w"]
TARGET = "dropped"
COHORTS = ["2025-1", "2025-2"]

# 세 가지 설정 — **한 번에 한 가지만 바꿉니다.**
RECIPES = {
    "A_기본":   {"features": ALL_FEATURES, "class_weight": None,       "C": 1.0},
    "B_가중치": {"features": ALL_FEATURES, "class_weight": "balanced", "C": 1.0},
    "C_출석만": {"features": ATTEND_ONLY,  "class_weight": "balanced", "C": 1.0},
}

# ── 데이터 (lab2 와 같습니다. 손대지 않아도 됩니다) ────────────────
files = [SRC / f"trainees_{c}.csv" for c in COHORTS]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates()
df[ALL_FEATURES] = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())

h = hashlib.sha256()
for f in files:
    h.update(f.read_bytes())
DATA_SHA = h.hexdigest()

# ── 기록 저장소 (여기까지는 되어 있습니다) ─────────────────────────
mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
mlflow.set_experiment("중도탈락위험_설정비교")
client = MlflowClient()


def run_one(name, cfg):
    feats = cfg["features"]
    X_tr, X_va, y_tr, y_va = train_test_split(
        df[feats], df[TARGET], test_size=0.25, random_state=SEED, stratify=df[TARGET])

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=cfg["C"],
                                   class_weight=cfg["class_weight"],
                                   random_state=SEED)),
    ])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    proba = model.predict_proba(X_va)[:, 1]
    metrics = {
        "accuracy":  round(float(accuracy_score(y_va, pred)), 4),
        "recall":    round(float(recall_score(y_va, pred)), 4),
        "precision": round(float(precision_score(y_va, pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_va, pred)), 4),
        "auc":       round(float(roc_auc_score(y_va, proba)), 4),
    }

    with mlflow.start_run(run_name=name) as run:
        
        # ── TODO 1 ─ 설정을 남기십시오 ─────────────────────────
        # 점검기가 보는 것: 기록마다 설정이 있고, 설정이 **서로 달라야** 합니다.
        # 남길 만한 것: recipe · features · n_features · class_weight · C ·
        #               seed · cohorts · data_sha256
        # mlflow.log_params({...})
            
        mlflow.log_params({
            "recipe": name,
            "features": ",".join(feats),
            "n_features": len(feats),
            "class_weight": str(cfg["class_weight"]),
            "C": cfg["C"],
            "seed": SEED,
            "data_sha256": DATA_SHA[:16],
        })
        
    
        # ── TODO 2 ─ 지표를 남기십시오 ─────────────────────────
        # metrics 딕셔너리가 이미 위에 있습니다.
        # mlflow.log_metrics(...)
        mlflow.log_metrics(metrics)
    
      

        # ── TODO 3 ─ 모델 자체도 같이 남기고, 이름에 등록하십시오 ────
        # 기록만 있고 물건이 없으면 꺼내 쓸 수 없습니다.
        # mlflow.sklearn.log_model(model, name="model", registered_model_name=...)
        mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME)
        return {"name": name, "run_id": run.info.run_id, **metrics}


results = [run_one(n, c) for n, c in RECIPES.items()]

# ── 고르기 ────────────────────────────────────────────────
# ⚠ 지금은 **정확도**로 고르게 되어 있습니다. 그대로 두어도 점검기는 통과합니다.
#    그런데 이 모델의 쓰임을 생각해 보십시오 — 이탈할 사람을 놓치는 것과
#    안 나갈 사람을 한 번 더 상담하는 것, 어느 쪽이 더 비쌉니까?
#    기준을 먼저 정하고 표를 보십시오.
best = max(results, key=lambda r: r["accuracy"])

# ── TODO 4 ─ 고른 것에 production 별칭을 거십시오 ──────────────
# 등록된 버전 목록에서 best["run_id"] 와 이어지는 버전을 찾아
# client.set_registered_model_alias(MODEL_NAME, "production", 그_버전) 를 부릅니다.
# versions = client.search_model_versions(f"name='{MODEL_NAME}'")
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
best_version = next(v.version for v in versions if v.run_id == best["run_id"])
client.set_registered_model_alias(MODEL_NAME, "production", best_version)


print(f"production → v{best_version}")
head = "| 설정 | 정확도 | 재현율 | 정밀도 | F1 | AUC |\n|---|---|---|---|---|---|\n"
rows = "".join(
    f"| {r['name']} | {r['accuracy']} | {r['recall']} | {r['precision']} | {r['f1']} | {r['auc']} |\n"
    for r in results)
print("\n" + head + rows)
print(f"고른 것: {best['name']}\n")

# ── TODO 5 ─ 실험비교.md 를 쓰십시오 (이 스크립트 밖, 손으로) ──────
# 위 표를 옮겨 붙이고, **왜 그것을 골랐는지**를 세 문장 안에 적습니다.
# 표만 있고 이유가 없으면 점검기가 잡습니다.
