"""중도탈락 위험 예측 — 추론 API (실습)

    python 준비.py                       ← 처음 한 번만. 모델을 등록해 둡니다
    uvicorn serve:app --reload --port 8000
    python ../검증/contract-test.py .

지금 이 파일은 **돌아갑니다.** 답도 줍니다. 그런데 쓸 수가 없습니다.
  · 버전 번호가 코드에 박혀 있어 모델을 갈아 끼울 때마다 이 파일을 고쳐야 합니다
  · 지금 누가 답하고 있는지 밖에서 알 방법이 없습니다
  · 이상한 값을 넣어도 그대로 받아 500으로 터집니다

할 일은 넷입니다.
  ① 모델을 **별칭**으로 부른다        버전을 박지 않는다
  ② /model 을 연다                  지금 누가 답하는지 밝힌다
  ③ 입력을 검증한다                  있을 수 없는 값은 4xx 로 막는다
  ④ 답에 버전을 같이 싣는다           「그때 그 답」을 추적할 끈

찾아볼 것: MlflowClient().get_model_version_by_alias · pydantic BaseModel · Field(ge=, le=)
"""
import pathlib
import pandas as pd
import mlflow
from mlflow import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel, Field

HERE = pathlib.Path(__file__).resolve().parent
DB = HERE / "mlflow.db"
MODEL_NAME = "dropout-risk"
ALIAS = "production"
THRESHOLD = 0.5

ALL_FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate",
                "assign_late_rate", "consult_count", "prior_experience"]

mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
client = MlflowClient()

# ── TODO 1 ─ 모델을 별칭으로 부르십시오 ───────────────────────
# 지금은 **버전 1이 박혀** 있습니다. 옛 모델(출석·지각 둘만 쓰는)입니다.
#   · 별칭이 가리키는 버전을 찾고  → client.get_model_version_by_alias(...)
#   · 그 별칭으로 모델을 불러오고  → f"models:/{MODEL_NAME}@{ALIAS}"
#   · 그 버전이 실제로 쓰는 특징 목록도 함께 (버전 태그 "features" 에 있습니다)
MODEL = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")
USES = ["attend_rate_4w", "late_count_4w"]

app = FastAPI(title="중도탈락 위험 예측")


# ── TODO 3 ─ 입력을 검증하십시오 ─────────────────────────────
# 지금은 받은 것을 그대로 믿습니다. 출석률 1.5 도, 항목이 빠진 것도 받습니다.
# pydantic 으로 **받는 모양 자체를 못 박으십시오.**
#   class Trainee(BaseModel):
#       attend_rate_4w: float = Field(ge=0, le=1)
#       ...
# 비율은 0~1, 횟수는 0 이상. 여섯 항목을 **전부** 받습니다 —
# 지금 모델이 그중 둘만 쓰더라도 **바깥과의 약속은 고정**입니다.


@app.get("/health")
def health():
    return {"status": "ok"}


# ── TODO 2 ─ GET /model 을 여십시오 ─────────────────────────
# 밖에서 「지금 누가 답하고 있나」를 물을 수 있어야 합니다.
# 돌려줄 것: 이름 · 버전 · 별칭 · 지금 쓰는 특징 · 임계값


@app.post("/predict")
def predict(t: dict):
    row = pd.DataFrame([t])[USES]
    risk = float(MODEL.predict_proba(row)[0][1])
    # ── TODO 4 ─ 답에 버전을 같이 실으십시오 ────────────────
    # 석 달 뒤 「이 답은 어느 모델이 준 겁니까」라는 질문이 옵니다.
    return {
        "risk": round(risk, 4),
        "label": int(risk >= THRESHOLD),
        "threshold": THRESHOLD,
    }
