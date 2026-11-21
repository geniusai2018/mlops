"""그림자 배포 점검.  사용법:  python 심화/checks/check-shadow.py [작업폴더] [기록저장소 폴더]     (lab6_rollback 폴더에서)

작업폴더(기본 심화/starter)의 shadow.py 를 도전자 v3 으로 두 번 돌립니다 — 평소대로 한 번, 도전자를 고장 낸 채 한 번.
기록저장소(기본 시작코드)는 mlflow.db 가 있는 폴더입니다(준비.py 를 돌린 곳).
"""
import json
import os
import pathlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
for k, v in (("MLFLOW_DISABLE_TELEMETRY", "true"), ("DO_NOT_TRACK", "1"), ("MLFLOW_DISABLE_AGENT_HINT", "1")):
    os.environ.setdefault(k, v)
import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

LAB = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else LAB / "심화" / "starter").resolve()
STORE = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else LAB / "시작코드").resolve()
NAME, N = "dropout-risk", 80
ok = bad = 0
out = []


def check(cid, cond, detail=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + detail if detail else ''}")


def run(extra=None):
    for f in ("shadow_log.jsonl", "shadow_summary.json"):
        (ROOT / f).unlink(missing_ok=True)
    r = subprocess.run([sys.executable, str(ROOT / "shadow.py"), "3", str(STORE)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8", **(extra or {})})
    log = ROOT / "shadow_log.jsonl"
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    s = ROOT / "shadow_summary.json"
    return r, rows, (json.loads(s.read_text(encoding="utf-8")) if s.exists() else {})


if not (STORE / "mlflow.db").exists():
    print(f"\n{STORE} 에 mlflow.db 가 없습니다. 그 폴더에서 `python 준비.py` 를 먼저 돌리십시오.\n")
    sys.exit(2)
mlflow.set_tracking_uri(f"sqlite:///{(STORE / 'mlflow.db').as_posix()}")
client = MlflowClient()
before = client.get_model_version_by_alias(NAME, "production").version
try:  # 앞서 걸어 둔 도전자 별칭이 있으면 떼고 시작한다
    client.delete_registered_model_alias(NAME, "challenger")
except Exception:
    pass

r, rows, summary = run()
try:
    chall = client.get_model_version_by_alias(NAME, "challenger").version
except Exception:
    chall = None
after = client.get_model_version_by_alias(NAME, "production").version

check("H1  challenger 별칭이 v3 에 걸렸다", str(chall) == "3", f"지금 challenger = {chall}" + (f" · {r.stderr.strip().splitlines()[-1][:90]}" if r.returncode and r.stderr.strip() else ""))
check("H2  production 별칭은 안 움직였다", str(before) == str(after), f"v{before} → v{after}")
both = [x for x in rows if x.get("production") and x.get("challenger")]
check(f"H3  요청 {N}건마다 두 모델의 답이 한 줄에 적혔다", len(rows) == N and len(both) == N, f"{len(rows)}줄 · 둘 다 있는 줄 {len(both)}")
mine = round(sum(x["production"]["label"] == x["challenger"]["label"] for x in both) / len(both), 4) if both else None
check("H4  요약의 일치율이 기록에서 다시 세어도 같다", mine is not None and summary.get("agreement") == mine,
      f"요약 {summary.get('agreement')} · 다시 센 값 {mine}")
p, c = summary.get("production") or {}, summary.get("challenger") or {}
check("H5  정답이 온 뒤의 성적이 두 모델 다 있다", all(isinstance(d.get(k), (int, float)) for d in (p, c) for k in ("recall", "precision")),
      f"운영 {p} · 도전자 {c}" if p and c else "")

r2, rows2, summary2 = run({"SHADOW_BREAK": "1"})
served = [x for x in rows2 if x.get("production")]
check("H6  도전자가 터져도 운영 모델은 80건 다 답한다",
      r2.returncode == 0 and len(served) == N and summary2.get("challenger_errors") == N,
      f"종료코드 {r2.returncode} · 운영 답 {len(served)}건 · 도전자 실패 {summary2.get('challenger_errors')}건"
      + (f" · {r2.stderr.strip().splitlines()[-1][:80]}" if r2.returncode and r2.stderr.strip() else ""))
run()  # 정상 상태의 기록으로 되돌려 둔다

print("\n" + "\n".join(out) + f"\n\n통과 {ok} · 실패 {bad}\n")
sys.exit(1 if bad else 0)
