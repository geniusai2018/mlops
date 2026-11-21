"""승격·되돌리기 점검.  사용법:  python 검증/check-rollback.py [작업폴더]

무엇을 보는가 — **올릴 것만 올라가고, 언제든 돌아올 수 있는가.**
  점검기가 직접 시나리오를 돌립니다. 불량 모델을 올려 보고, 좋은 모델을 올리고,
  되돌리고, 그 사이 API가 멀쩡한지까지 봅니다.

⚠ 이 점검기는 **작업폴더의 별칭을 실제로 움직입니다.** 끝나면 원래 자리로 돌려놓습니다.

기대하는 것
  작업폴더/promote.py     python promote.py <버전> [사유]   기준 미달이면 종료코드 0이 아님
  작업폴더/rollback.py    python rollback.py                직전으로 되돌린다
  작업폴더/승격이력.jsonl  올린 것도 되돌린 것도 한 줄씩
"""
import os, json, pathlib, subprocess, sys, warnings
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "1")
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "시작코드").resolve()
DB = ROOT / "mlflow.db"
LOG = ROOT / "승격이력.jsonl"
MODEL_NAME, ALIAS = "dropout-risk", "production"
SERVE = ROOT.parent / "공용" / "serve.py"

ok = bad = 0
out = []

def check(cid, cond, why="", info=""):
    """why 는 실패했을 때만, info 는 언제나 보여 준다."""
    global ok, bad
    if cond: ok += 1
    else: bad += 1
    tail = info or ("" if cond else why)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + tail if tail else ''}")

def finish(restore=None):
    if restore:
        try:
            client.set_registered_model_alias(MODEL_NAME, ALIAS, restore)
        except Exception:
            pass
    print(f"\n점검 대상: {ROOT}\n" + "\n".join(out))
    print(f"\n통과 {ok} · 실패 {bad}\n")
    if bad:
        print("되돌릴 수 없는 배포는 배포가 아니라 도박입니다.\n")
    sys.exit(1 if bad else 0)

def run(script, *args):
    return subprocess.run([sys.executable, script, *[str(a) for a in args]], cwd=ROOT,
                          capture_output=True, text=True, timeout=300,
                          encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})

def lines():
    if not LOG.exists():
        return []
    return [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]

check("B0  promote.py · rollback.py 가 있다",
      (ROOT / "promote.py").exists() and (ROOT / "rollback.py").exists(),
      why="둘 다 있어야 합니다")
check("B1  기록 저장소(mlflow.db)가 있다", DB.exists(), why="준비.py 를 먼저 돌리십시오")
if bad: finish()

import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
client = MlflowClient()

versions = {(v.tags or {}).get("label"): v.version
            for v in client.search_model_versions(f"name='{MODEL_NAME}'")}
BAD_V = versions.get("불량_라벨섞임")
GOOD_V = versions.get("재학습_새기수만")
check("B2  준비된 네 버전이 모두 있다", BAD_V and GOOD_V and len(versions) >= 4,
      why=f"찾은 것: {sorted(k for k in versions if k)}", info=f"{len(versions)}개")
if bad: finish()

START = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
n0 = len(lines())

# ── 관문이 막는가 ──────────────────────────────────────────
r = run("promote.py", BAD_V, "점검기: 불량 모델 승격 시도")
after_bad = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
check("B3  **불량 모델은 승격을 거부**한다 (종료코드 ≠ 0)", r.returncode != 0,
      why=f"종료코드 {r.returncode} — 관문이 통과시켰다. 기준을 확인하라",
      info=(f"v{BAD_V} 거부됨" if r.returncode != 0 else ""))
check("B4  거부되면 별칭이 **안 움직인다**", after_bad == START,
      why=f"production 이 v{START} → v{after_bad} 로 움직였다")

# ── 좋은 것은 올라가는가 ────────────────────────────────────
r = run("promote.py", GOOD_V, "점검기: 재학습 모델 승격")
after_good = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
check("B5  기준을 넘는 모델은 승격된다", r.returncode == 0 and after_good == GOOD_V,
      why=f"종료코드 {r.returncode} · production = v{after_good}",
      info=f"v{START} → v{after_good}")

hist = lines()
promo = [h for h in hist if h.get("action") == "promote"]
check("B6  승격이 **이력에 남는다** (무엇→무엇 · 왜)",
      bool(promo) and str(promo[-1].get("to")) == str(GOOD_V)
      and promo[-1].get("from") is not None and bool(promo[-1].get("why")),
      why="from · to · why 가 한 줄에 같이 있어야 합니다",
      info=f"v{promo[-1].get('from')} → v{promo[-1].get('to')}" if promo else "")

# ── 되돌아오는가 ──────────────────────────────────────────
r = run("rollback.py")
back = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
check("B7  **버전을 대지 않아도** 직전으로 되돌아온다", r.returncode == 0 and back == START,
      why=f"production = v{back} (직전은 v{START}) — 직전 자리를 이력에서 찾으십시오",
      info=f"v{GOOD_V} → v{back}")

rb = [h for h in lines() if h.get("action") == "rollback"]
check("B8  되돌린 것도 이력에 남는다", bool(rb) and str(rb[-1].get("to")) == str(START),
      why="되돌리기도 사건입니다. 남기지 않으면 다음 사람이 처음부터 조사합니다")

# ── 부르는 쪽은 멀쩡한가 ────────────────────────────────────
served = None
if SERVE.exists():
    try:
        import importlib.util
        os.chdir(ROOT)
        spec = importlib.util.spec_from_file_location("serve_check", SERVE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["serve_check"] = mod
        spec.loader.exec_module(mod)
        from fastapi.testclient import TestClient
        c = TestClient(mod.app, raise_server_exceptions=False)
        served = str(c.get("/model").json().get("version"))
        good = c.post("/predict", json={"attend_rate_4w": 0.62, "late_count_4w": 5,
                                        "assign_submit_rate": 0.41, "assign_late_rate": 0.55,
                                        "consult_count": 0, "prior_experience": 0})
        api_ok = good.status_code == 200 and served == str(back)
    except Exception as e:
        api_ok, served = False, f"{type(e).__name__}: {e}"
else:
    api_ok = False

check("B9  되돌린 뒤 API가 **그대로 돌아간다** (고친 코드 없음)", api_ok,
      why=f"serve.py 가 v{served} 를 보고 있다 (별칭은 v{back})",
      info=f"serve.py → v{served}" if api_ok else "")

check("B10 승격·되돌리기가 **코드 수정 없이** 됐다",
      len(lines()) > n0 and (ROOT / "promote.py").exists(),
      why="이력이 늘지 않았다")

finish(restore=START)
