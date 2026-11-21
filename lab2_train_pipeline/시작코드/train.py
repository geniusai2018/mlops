"""중도탈락 위험 예측 — 학습 스크립트 (실습)

    python train.py 출력폴더
      → 출력폴더/predictions.csv   검증셋 예측
      → 출력폴더/RUN.json          씨앗값·설정·데이터 버전·성능 지표

목표는 하나입니다 — **두 번 돌리면 같은 것이 나오게.**
    python ../검증/check-repro.py .
"""
import json, pathlib, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

HERE = pathlib.Path(__file__).parent
SRC = HERE.parent.parent / "공용데이터"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "run")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate",
            "assign_late_rate", "consult_count", "prior_experience"]
TARGET = "dropped"

# ── 데이터 ─────────────────────────────────────────────
df = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]],
               ignore_index=True)
df = df.drop_duplicates()
df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

# ── 나누기 ─────────────────────────────────────────────
# TODO 1 — 여기가 매번 달라지는 자리입니다. 무엇을 고정해야 합니까?
X_tr, X_va, y_tr, y_va = train_test_split(df[FEATURES], df[TARGET], test_size=0.25)

# ── 학습 ──────────────────────────────────────────────
# TODO 2 — 모델에도 고정해야 할 것이 있습니다
model = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_tr, y_tr)

# ── 평가 ──────────────────────────────────────────────
pred = model.predict(X_va)
proba = model.predict_proba(X_va)[:, 1]
metrics = {
    "recall": round(float(recall_score(y_va, pred)), 4),
    "precision": round(float(precision_score(y_va, pred, zero_division=0)), 4),
    "f1": round(float(f1_score(y_va, pred)), 4),
    "auc": round(float(roc_auc_score(y_va, proba)), 4),
}

# ── 산출물 ────────────────────────────────────────────
pd.DataFrame({"y_true": y_va.values, "y_pred": pred, "proba": proba.round(6)}) \
  .to_csv(OUT / "predictions.csv", index=False)

# TODO 3 — RUN.json 에 무엇을 남겨야 「이 모델은 무엇으로 배웠는가」에 답할 수 있습니까?
#   점검기가 보는 것: seed · params · data_sha256(또는 data_version) · metrics
record = {
    "metrics": metrics,
}
(OUT / "RUN.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False))
