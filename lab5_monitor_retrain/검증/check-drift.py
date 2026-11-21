"""드리프트 감시·재학습 판단 점검.  사용법:  python 검증/check-drift.py [작업폴더]

무엇을 보는가 — **변한 것을 변했다고 하고, 안 변한 것을 가만두는가.**
  경보가 안 울리면 사고를 놓치고, 아무 때나 울리면 아무도 안 봅니다.
  그리고 하나 더 — **재학습한 모델을 제멋대로 올리지 않는가.**

기대하는 것
  작업폴더/drift_report.json   무엇이 변했나 · 얼마나 나빠졌나 · 그래서 어떻게 할 건가
  작업폴더/mlflow.db           재학습 모델이 **새 버전으로 등록**되어 있다
                               production 별칭은 **그대로** — 승격은 사람이 한다
"""
import json, pathlib, sys, warnings
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "시작코드").resolve()
REPORT = ROOT / "drift_report.json"
DB = ROOT / "mlflow.db"
MODEL_NAME = "dropout-risk"

# 이 자료에서 실제로 달라진 열과 그대로인 열 (비대면 전환)
MUST_FLAG = ["attend_rate_4w", "late_count_4w"]
MUST_NOT_FLAG = ["consult_count", "prior_experience", "assign_late_rate"]
FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate",
            "assign_late_rate", "consult_count", "prior_experience"]

ok = bad = 0
out = []

def check(cid, cond, why="", info=""):
    """why 는 실패했을 때만, info 는 언제나 보여 준다."""
    global ok, bad
    if cond: ok += 1
    else: bad += 1
    tail = info or ("" if cond else why)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + tail if tail else ''}")

def report_and_exit():
    print(f"\n점검 대상: {ROOT}\n" + "\n".join(out))
    print(f"\n통과 {ok} · 실패 {bad}\n")
    if bad:
        print("울리지 않는 경보와 아무 때나 울리는 경보는 둘 다 없는 것과 같습니다.\n")
    sys.exit(1 if bad else 0)

check("M0  drift_report.json 이 있다", REPORT.exists(), why=f"{REPORT} 가 없다")
if not REPORT.exists(): report_and_exit()

try:
    R = json.loads(REPORT.read_text(encoding="utf-8"))
except Exception as e:
    check("M0b 읽을 수 있는 JSON 이다", False, why=str(e)); report_and_exit()

cols = R.get("features") or {}
have = [f for f in FEATURES if f in cols and "psi" in (cols.get(f) or {})]
check("M1  여섯 특징을 모두 견주었다 (psi)", len(have) == len(FEATURES),
      why=f"빠진 열: {[f for f in FEATURES if f not in have]}", info=f"{len(have)}/6")

def flagged(f):
    c = cols.get(f) or {}
    return bool(c.get("drift")) or c.get("state") in ("이상", "drift", "alert")

missed = [f for f in MUST_FLAG if not flagged(f)]
check("M2  **변한 열**을 이상으로 잡았다 (출석률 · 지각)", not missed,
      why=f"놓친 열: {missed} — 비대면 전환으로 확실히 달라진 값이다",
      info="출석률 · 지각" if not missed else "")

false_alarm = [f for f in MUST_NOT_FLAG if flagged(f)]
check("M3  **안 변한 열**은 가만두었다 (거짓 경보 없음)", not false_alarm,
      why=f"잘못 울린 열: {false_alarm} — 아무 때나 울리는 경보는 아무도 안 본다")

perf = R.get("performance") or {}
on_base = perf.get("on_baseline") or {}
on_target = perf.get("on_target") or {}
both = "recall" in on_base and "recall" in on_target
check("M4  운영 모델 성능을 옛 기수·새 기수 둘 다 쟀다", both,
      why="한쪽만 있으면 「나빠졌다」고 말할 수 없다",
      info=f"{on_base.get('recall')} → {on_target.get('recall')}" if both else "")

check("M5  성능이 실제로 떨어졌음이 보인다", both and on_target["recall"] < on_base["recall"] - 0.2,
      why="새 기수에서 재현율이 크게 떨어져야 정상이다. 무엇을 쟀는지 다시 보라")

dec = R.get("decision") or {}
check("M6  재학습 판단과 **근거**가 남아 있다",
      dec.get("retrain") is True and bool(dec.get("reasons")),
      why="무엇 때문에 다시 배우기로 했는지가 없다",
      info=str(dec.get("reasons", [""])[0])[:50] if dec.get("reasons") else "")

# ── 등록소 쪽 ─────────────────────────────────────────────
check("M7  기록 저장소(mlflow.db)가 있다", DB.exists(), why=f"{DB} 가 없다")
if not DB.exists(): report_and_exit()

import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
client = MlflowClient()

try:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
except Exception:
    versions = []
retrained = [v for v in versions if (v.tags or {}).get("trigger")]
check("M8  재학습 모델이 **새 버전으로 등록**되어 있다", bool(retrained),
      why="다시 배우기만 하고 등록하지 않으면 꺼내 쓸 수 없다",
      info=f"v{retrained[-1].version}" if retrained else "")

try:
    live = client.get_model_version_by_alias(MODEL_NAME, "production").version
except Exception:
    live = None
auto_promoted = bool(retrained) and live in {v.version for v in retrained}
check("M9  production 별칭을 **건드리지 않았다** (승격은 사람이)",
      live is not None and not auto_promoted,
      why="재학습 모델을 스스로 올렸다 — 자동으로 올리면 사고도 자동이 된다",
      info=f"production = v{live}" if live else "")

rt = R.get("retrained") or {}
after = (rt.get("metrics_on_target") or {}).get("recall")
check("M10 재학습 모델이 새 기수에서 더 낫다",
      after is not None and on_target.get("recall") is not None and after > on_target["recall"],
      why="새 자료로 배웠는데 나아지지 않았다면, 배운 자료를 다시 보라",
      info=f"{on_target.get('recall')} → {after}" if after is not None else "")

report_and_exit()
