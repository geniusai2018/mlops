"""중도탈락 위험 예측 — 추론 API (lab4 완성본 그대로 · 손대지 마십시오)

    uvicorn serve:app --reload --port 8000
    python ../검증/contract-test.py .

이 파일이 지키는 약속 셋
  GET  /health    살아 있는가
  GET  /model     지금 누가 답하고 있는가 (이름 · 버전 · 별칭)
  POST /predict   위험도 한 건

**바깥과의 약속은 고정하고, 안쪽 모델은 갈아 끼웁니다.**
입력은 언제나 여섯 항목을 받습니다. 지금 걸린 모델이 그중 둘만 쓰더라도
부르는 쪽 코드는 고치지 않습니다 — 그래야 되돌리기(lab6)가 사고 없이 됩니다.
"""
import pathlib
import pandas as pd
import mlflow
from mlflow import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel, Field

HERE = pathlib.Path(__file__).resolve().parent
# lab6에서는 **작업 폴더의** 기록 저장소를 봅니다 (돌린 자리 기준).
DB = pathlib.Path.cwd() / "mlflow.db"
MODEL_NAME = "dropout-risk"
ALIAS = "production"
THRESHOLD = 0.5

ALL_FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate",
                "assign_late_rate", "consult_count", "prior_experience"]

mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
client = MlflowClient()

# ── 모델은 **별칭으로** 불러옵니다 ─────────────────────────────
# 버전 번호를 여기 박으면, 버전을 바꿀 때마다 이 파일을 고쳐야 합니다.
mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
MODEL = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
USES = (mv.tags.get("features") or ",".join(ALL_FEATURES)).split(",")

app = FastAPI(title="중도탈락 위험 예측")


class Trainee(BaseModel):
    """받는 것은 언제나 여섯 항목. 범위를 벗어난 값은 여기서 막힙니다."""
    attend_rate_4w: float = Field(ge=0, le=1)
    late_count_4w: int = Field(ge=0, le=100)
    assign_submit_rate: float = Field(ge=0, le=1)
    assign_late_rate: float = Field(ge=0, le=1)
    consult_count: float = Field(ge=0, le=100)
    prior_experience: int = Field(ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    return {"name": MODEL_NAME, "version": mv.version, "alias": ALIAS,
            "features_used": USES, "threshold": THRESHOLD}


@app.post("/predict")
def predict(t: Trainee):
    row = pd.DataFrame([t.model_dump()])[USES]      # 지금 모델이 쓰는 열만 골라 넣는다
    risk = float(MODEL.predict_proba(row)[0][1])
    return {
        "risk": round(risk, 4),
        "label": int(risk >= THRESHOLD),
        "threshold": THRESHOLD,
        # **무엇이 답했는지를 답과 함께 돌려줍니다.** 나중에 「그때 그 답」을 추적하는 유일한 끈입니다.
        "model": MODEL_NAME,
        "version": mv.version,
    }
