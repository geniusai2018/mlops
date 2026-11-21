"""되돌리기 — 직전으로, 코드를 고치지 않고 (실습)

    python rollback.py [버전]

지금 이 파일은 **버전을 대야만** 되돌립니다.
사고가 난 한복판에서 「어느 버전이었지」를 찾는 일이 바로 그 5분입니다.

할 일은 둘입니다.
  ① 버전을 안 줘도 **직전으로** 돌아간다   승격이력에서 찾습니다
  ② 되돌린 것도 이력에 남긴다             되돌리기도 사건입니다

되돌리기에서 중요한 것은 빠르기가 아니라 **망설이지 않음**입니다.
무엇으로 돌아가야 하는지 이미 적혀 있으면, 판단할 것이 없습니다.
"""
import os, sys, json, pathlib, datetime
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "1")
import mlflow
from mlflow import MlflowClient

HERE = pathlib.Path(__file__).resolve().parent
DB = HERE / "mlflow.db"
LOG = HERE / "승격이력.jsonl"
MODEL_NAME, ALIAS = "dropout-risk", "production"


def history():
    """승격이력을 줄 단위로 읽어 옵니다. (만들어 두었습니다)"""
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    target = str(sys.argv[1]) if len(sys.argv) > 1 else None

    mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
    client = MlflowClient()
    current = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version

    # ── TODO 1 ─ 버전을 안 줬을 때 직전을 찾으십시오 ───────────────
    # history() 에서 별칭을 움직인 줄(action 이 promote 나 rollback)을 찾아
    # 그 줄의 "from" 이 돌아갈 자리입니다.
    # 이력이 비어 있으면 **되돌릴 자리를 모른다**고 말하고 멈추십시오 —
    # 그것이 기록을 남기는 이유입니다.
    if target is None:
        print("사용법: python rollback.py <버전>")
        return 2

    client.set_registered_model_alias(MODEL_NAME, ALIAS, target)

    # ── TODO 2 ─ 되돌린 것도 이력에 남기십시오 ──────────────────
    # 점검기가 보는 것: at · action("rollback") · from · to

    print(f"\n  되돌림 — v{current} → v{target}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
