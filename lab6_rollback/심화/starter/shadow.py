"""그림자 배포 — 도전자 모델을 운영 모델 옆에 세운다 (심화 시작 코드)

    python shadow.py <도전자 버전> [기록저장소 폴더]        예) python shadow.py 3 ../../시작코드

운영 모델(@production)이 평소처럼 답하고, 도전자(@challenger)는 **같은 입력에 답만 적어 둡니다.**
바깥으로 나가는 답은 언제나 운영 모델의 것입니다. 도전자가 터져도 운영은 멈추지 않습니다.
"""
import json
import os
import pathlib
import sys

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from sklearn.metrics import precision_score, recall_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 20261121
HERE = pathlib.Path(__file__).resolve().parent
SRC = next(p / "공용데이터" for p in HERE.parents if (p / "공용데이터").is_dir())
STORE = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else HERE.parents[1] / "시작코드").resolve()
LOG, SUMMARY = HERE / "shadow_log.jsonl", HERE / "shadow_summary.json"
MODEL_NAME = "dropout-risk"
FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate", "assign_late_rate", "consult_count", "prior_experience"]


class Broken:
    """도전자가 고장 난 상황 흉내 (환경 변수 SHADOW_BREAK=1)."""

    def predict_proba(self, _):
        raise RuntimeError("challenger is down")


def todays_requests():
    """오늘 들어온 요청 — 지금 세상(2026-2)의 평가분 80건. 정답은 **나중에** 온다."""
    old = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]], ignore_index=True).drop_duplicates()
    new = pd.read_csv(SRC / "trainees_2026-2.csv")
    new[FEATURES] = new[FEATURES].fillna(old[FEATURES].median())
    _, X, _, y = train_test_split(new[FEATURES], new["dropped"], test_size=0.25, random_state=SEED, stratify=new["dropped"])
    return X.reset_index(drop=True), y.reset_index(drop=True)


def load(client, ref):
    mv = client.get_model_version_by_alias(MODEL_NAME, ref)
    uses = (mv.tags.get("features") or ",".join(FEATURES)).split(",")
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ref}"), uses, mv.version


def one(model, uses, row):
    risk = float(model.predict_proba(row[uses])[0][1])
    return {"risk": round(risk, 4), "label": int(risk >= 0.5)}


def answer(row, prod, chall, log):
    """한 건에 답한다. 돌려주는 것은 운영 모델의 답뿐이다."""
    p_model, p_uses, p_ver = prod
    c_model, c_uses, c_ver = chall
    mine = {"version": p_ver, **one(p_model, p_uses, row)}
    # TODO 2 — 도전자에게도 같은 입력을 주고, 두 답을 한 줄로 적으십시오.
    #   {"production": {version, risk, label}, "challenger": {version, risk, label} 또는 None, "error": 예외 이름 또는 None}
    #   **도전자가 터져도 이 함수는 운영 모델의 답을 돌려줘야 합니다.**
    return mine


def main():
    if len(sys.argv) < 2:
        print("사용법: python shadow.py <도전자 버전> [기록저장소 폴더]")
        return 2
    mlflow.set_tracking_uri(f"sqlite:///{(STORE / 'mlflow.db').as_posix()}")
    client = MlflowClient()
    before = client.get_model_version_by_alias(MODEL_NAME, "production").version
    if str(sys.argv[1]) == str(before):
        print("도전자가 운영 모델과 같은 버전입니다.")
        return 1
    # TODO 1 — 받은 버전에 challenger 별칭을 거십시오. production 은 건드리지 않습니다.

    prod, chall = load(client, "production"), load(client, "challenger")
    if os.environ.get("SHADOW_BREAK"):
        chall = (Broken(), chall[1], chall[2])

    X, y = todays_requests()
    with LOG.open("w", encoding="utf-8") as log:
        served = [answer(X.iloc[[i]], prod, chall, log) for i in range(len(X))]

    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines()]
    both = [(r["production"]["label"], r["challenger"]["label"]) for r in rows if r["challenger"]]
    errors = sum(1 for r in rows if r["error"])

    def score(labels):
        return {"recall": round(float(recall_score(y, labels)), 4), "precision": round(float(precision_score(y, labels, zero_division=0)), 4)}

    summary = {
        "rows": len(rows), "served": len(served),
        "production_version": prod[2], "challenger_version": chall[2],
        "agreement": None,   # TODO 3 — 두 모델의 판정(label)이 같았던 비율. 도전자가 답한 줄만 셉니다
        "challenger_errors": errors,
        # 정답이 도착한 뒤에야 채울 수 있는 칸
        "production": score([r["production"]["label"] for r in rows]),
        "challenger": score([b for _, b in both]) if len(both) == len(rows) else None,
        "production_alias_after": client.get_model_version_by_alias(MODEL_NAME, "production").version,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
