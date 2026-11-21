"""실험 기록·모델 등록 점검.  사용법:  python 검증/check-registry.py [작업폴더]

무엇을 보는가 — **어느 것이 더 나은지 말할 수 있는가, 그리고 그것을 꺼내 쓸 수 있는가.**
  실험이 기록되지 않으면 비교할 수 없고, 비교하지 않으면 무엇을 배포할지 정할 근거가 없습니다.

기대하는 것
  작업폴더/mlflow.db          기록 저장소 (sqlite)
  등록 모델 이름 dropout-risk · 별칭 production 이 한 버전에 걸려 있다
  작업폴더/실험비교.md         사람이 읽는 비교표와 선택 이유
"""
import pathlib, sys, warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "시작코드").resolve()
DB = ROOT / "mlflow.db"
MODEL_NAME = "dropout-risk"
ok = bad = 0
out = []

def check(cid, cond, why="", info=""):
    global ok, bad
    if cond: ok += 1
    else: bad += 1
    tail = info or ("" if cond else why)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + tail if tail else ''}")

check("E0  기록 저장소(mlflow.db)가 있다", DB.exists(), why=f"{DB} 가 없다")
if not DB.exists():
    print("\n" + "\n".join(out)); print(f"\n통과 {ok} · 실패 {bad}\n"); sys.exit(1)

import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
client = MlflowClient()

runs = []
for exp in client.search_experiments():
    runs += client.search_runs([exp.experiment_id], max_results=100)
finished = [r for r in runs if r.info.status == "FINISHED"]

check("E1  실험 기록이 3건 이상", len(finished) >= 3, info=f"{len(finished)}건")

# 같은 이름으로 다시 돌린 기록은 **가장 최근 것**으로 판정한다
# (TODO 를 고치기 전에 한 번 돌려 본 빈 기록 때문에 영영 실패하지 않게 — 2026-11-15)
latest = {}
for r in sorted(finished, key=lambda r: r.info.start_time or 0):
    latest[r.data.tags.get("mlflow.runName", r.info.run_id)] = r
judged = list(latest.values())
lack = [r.info.run_id[:8] for r in judged if not r.data.params or not r.data.metrics]
check("E2  기록마다 설정과 지표가 남아 있다", finished and not lack,
      why=f"빠진 기록: {lack}")

sigs = {tuple(sorted(r.data.params.items())) for r in judged}
check("E3  실험끼리 설정이 서로 다르다", len(sigs) >= 3,
      why=f"서로 다른 설정 {len(sigs)}가지 — 같은 것을 되풀이하면 비교가 아니다")

try:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
except Exception:
    versions = []
check("E4  모델이 등록되어 있다", len(versions) >= 1,
      why=f"등록 모델 '{MODEL_NAME}' 을 찾지 못했다", info=f"{len(versions)}개 버전" if versions else "")

run_ids = {r.info.run_id for r in runs}
linked = [v for v in versions if v.run_id in run_ids]
check("E5  등록된 모델이 실제 실험을 가리킨다", bool(linked),
      why="어느 실험에서 나온 모델인지 이어지지 않는다")

aliases = {}
try:
    rm = client.get_registered_model(MODEL_NAME)
    aliases = dict(rm.aliases or {})
except Exception:
    pass
check("E6  production 별칭이 한 버전에 걸려 있다", "production" in aliases,
      why=f"걸린 별칭: {sorted(aliases) or '없음'}",
      info=f"v{aliases['production']}" if "production" in aliases else "")

card = ROOT / "실험비교.md"
body = card.read_text(encoding="utf-8") if card.exists() else ""
rows = body.count("|") if body else 0
check("E7  실험비교.md 에 비교표와 **선택 이유**가 있다",
      card.exists() and rows >= 12 and any(k in body for k in ("이유", "고른", "선택")),
      why="없거나, 표만 있고 왜 그것을 골랐는지가 없다")

print(f"\n점검 대상: {ROOT}\n" + "\n".join(out))
print(f"\n통과 {ok} · 실패 {bad}\n")
if bad:
    print("비교할 수 없는 실험은 하지 않은 것과 같습니다.\n")
sys.exit(1 if bad else 0)
