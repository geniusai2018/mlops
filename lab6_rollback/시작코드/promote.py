"""승격 — 올릴 것만 올린다 (실습)

    python 준비.py                      ← 처음 한 번만
    python promote.py <버전> [사유]
    python ../검증/check-rollback.py .

지금 이 파일은 **시키는 대로 그냥 올립니다.** 묻지도 않고, 적지도 않습니다.
불량 모델을 대면 불량 모델이 운영에 올라갑니다.

할 일은 셋입니다.
  ① 관문        기준 미달이면 **올리지 않는다.** 종료코드도 0이 아니어야 합니다
  ② 기록        무엇을 무엇으로 바꿨는지, 왜 바꿨는지
  ③ 되돌릴 자리  직전 버전을 적어 둡니다. 그래야 rollback.py 가 찾을 수 있습니다

세 번째가 빠지면, 사고가 났을 때 **무엇으로 돌아가야 하는지부터 찾게** 됩니다.
그 5분이 사고를 30분으로 만듭니다.
"""
import os, sys, json, pathlib, datetime
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "1")
import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score

SEED = 20261121
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent.parent / "공용데이터"
DB = HERE / "mlflow.db"
LOG = HERE / "승격이력.jsonl"
MODEL_NAME, ALIAS = "dropout-risk", "production"

# 관문 기준 — **lab3 실험비교.md 에 적어 둔 그 조건입니다.** 그때 정하고 지금 씁니다.
RECALL_FLOOR, PRECISION_FLOOR = 0.60, 0.40

FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate",
            "assign_late_rate", "consult_count", "prior_experience"]
TARGET = "dropped"


def gate_data():
    """지금 세상(2026-2) 평가분. **승격 판정은 지금 세상에서 합니다.**
    (이 함수는 만들어 두었습니다)"""
    old = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]],
                    ignore_index=True).drop_duplicates()
    med = old[FEATURES].median()
    new = pd.read_csv(SRC / "trainees_2026-2.csv")
    new[FEATURES] = new[FEATURES].fillna(med)
    _, Xva, _, yva = train_test_split(new[FEATURES], new[TARGET], test_size=0.25,
                                      random_state=SEED, stratify=new[TARGET])
    return Xva, yva


def main():
    if len(sys.argv) < 2:
        print("사용법: python promote.py <버전> [사유]"); return 2
    target = str(sys.argv[1])
    why = sys.argv[2] if len(sys.argv) > 2 else "(사유 없음)"

    mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
    client = MlflowClient()
    mv = client.get_model_version(MODEL_NAME, target)

    # ── TODO 1 ─ 관문을 세우십시오 ────────────────────────────
    # 올리려는 버전을 불러와 gate_data() 의 평가분으로 재고,
    # 재현율이 RECALL_FLOOR 보다 낮거나 정밀도가 PRECISION_FLOOR 보다 낮으면
    # **올리지 말고** 이유를 찍은 뒤 0이 아닌 값을 돌려주십시오.
    #   uses  = (mv.tags.get("features") or ",".join(FEATURES)).split(",")
    #   model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{target}")

    # ── TODO 2 ─ 되돌릴 자리를 먼저 집어 두십시오 ─────────────────
    # 별칭을 옮기기 **전에** 지금 걸려 있는 버전을 읽어 둡니다.
    # 옮기고 나서 읽으면 이미 늦습니다.
    #   current = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version

    client.set_registered_model_alias(MODEL_NAME, ALIAS, target)

    # ── TODO 3 ─ 이력을 남기십시오 ───────────────────────────
    # 승격이력.jsonl 에 **한 줄** 덧붙입니다. 점검기가 보는 것:
    #   at · action("promote") · from · to · why
    # 거부했을 때도 한 줄 남겨 두면, 「왜 안 올렸나」가 나중에 설명됩니다.

    print(f"\n  승격 — v{target} ({mv.tags.get('label', '')})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
