"""지연과 묶음 예측 점검.  사용법:  python 심화/checks/check-latency.py [작업폴더]     (lab4_serve_cicd 폴더에서)

작업폴더(기본 시작코드)의 serve.py 를 불러 직접 재 봅니다. mlflow.db 가 그 폴더에 있어야 합니다(준비.py).
"""
import importlib.util
import os
import pathlib
import statistics
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

LAB = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else LAB / "시작코드").resolve()
ok = bad = 0
out = []


def check(cid, cond, detail=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + detail if detail else ''}")


def done():
    print("\n" + "\n".join(out) + f"\n\n통과 {ok} · 실패 {bad}\n")
    sys.exit(1 if bad else 0)


try:
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("serve", ROOT / "serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    api = TestClient(mod.app)
except Exception as e:
    check("L0  serve.py 를 불러온다", False, f"{type(e).__name__}: {str(e)[:120]}")
    done()

one = {"attend_rate_4w": 0.62, "late_count_4w": 4, "assign_submit_rate": 0.55,
       "assign_late_rate": 0.4, "consult_count": 1, "prior_experience": 0}
rows = [{**one, "attend_rate_4w": round(0.4 + 0.003 * k, 3), "late_count_4w": k % 6} for k in range(200)]

r = api.post("/predict/batch", json={"items": rows})
body = r.json() if r.status_code == 200 else {}
results = body.get("results", [])
check("L1  POST /predict/batch 가 200 이고 보낸 수만큼 답한다", r.status_code == 200 and len(results) == 200,
      f"받은 값 {r.status_code} · {len(results)}건")
check("L2  묶음의 답에도 버전이 실린다", bool(body.get("version")))

singles = [api.post("/predict", json=x).json() for x in rows[:20]]
match = len(results) >= 20 and all(abs(a.get("risk", -1) - b.get("risk", -2)) < 1e-4 and a.get("label") == b.get("label")
                                   for a, b in zip(singles, results[:20]))
check("L3  묶음으로 보낸 답이 한 건씩 보낸 답과 같다", match)

empty, huge = api.post("/predict/batch", json={"items": []}), api.post("/predict/batch", json={"items": [one] * 1001})
check("L4  빈 묶음과 1001건은 막는다 (400·413·422)", empty.status_code in (400, 413, 422) and huge.status_code in (400, 413, 422),
      f"빈 묶음 {empty.status_code} · 1001건 {huge.status_code}")

if r.status_code == 200:
    times = []
    for x in rows:
        t = time.perf_counter()
        api.post("/predict", json=x)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    p50, p95 = statistics.median(times), times[int(len(times) * 0.95) - 1]
    batch_times = []
    for _ in range(3):
        t = time.perf_counter()
        api.post("/predict/batch", json={"items": rows})
        batch_times.append((time.perf_counter() - t) * 1000)
    batch_ms = min(batch_times)
    per_row = batch_ms / len(rows)
    check("L5  묶음이 한 건당 시간으로는 더 빠르다", per_row < p50,
          f"한 건씩 p50 {p50:.2f}ms · p95 {p95:.2f}ms / 묶음 200건 {batch_ms:.1f}ms = 건당 {per_row:.3f}ms")
else:
    check("L5  묶음이 한 건당 시간으로는 더 빠르다", False, "묶음 창구가 아직 없습니다")

c = subprocess.run([sys.executable, str(LAB / "검증" / "contract-test.py"), str(ROOT)], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
check("L6  실습 4의 계약 시험 10개가 그대로 통과한다", c.returncode == 0, "" if c.returncode == 0 else "검증/contract-test.py 를 돌려 보십시오")
done()
