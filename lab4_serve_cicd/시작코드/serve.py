"""중도탈락 위험 예측 API."""
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

mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
client = MlflowClient()
MV = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
MODEL = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
USES = MV.tags["features"].split(",")

app = FastAPI(title="중도탈락 위험 예측")


class Trainee(BaseModel):
    attend_rate_4w: float = Field(ge=0, le=1)
    late_count_4w: int = Field(ge=0)
    assign_submit_rate: float = Field(ge=0, le=1)
    assign_late_rate: float = Field(ge=0, le=1)
    consult_count: int = Field(ge=0)
    prior_experience: int = Field(ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    return {
        "name": MODEL_NAME,
        "version": MV.version,
        "alias": ALIAS,
        "features_used": USES,
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(t: Trainee):
    row = pd.DataFrame([t.model_dump()], columns=USES)
    risk = float(MODEL.predict_proba(row)[0][1])
    return {
        "risk": round(risk, 4),
        "label": int(risk >= THRESHOLD),
        "threshold": THRESHOLD,
        "model" : MODEL_NAME,
        "version": MV.version,
    }
