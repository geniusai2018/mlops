"""생성형 심화 점검.  사용법:  python checks/check-llmops.py [작업폴더]     (심화_생성형 폴더에서)

작업폴더(기본 starter)의 llmops.py 를 **임시 저장소**에서 돌려 봅니다. 여러분의 기록은 건드리지 않습니다.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
for k, v in (("MLFLOW_DISABLE_TELEMETRY", "true"), ("DO_NOT_TRACK", "1"), ("MLFLOW_DISABLE_AGENT_HINT", "1")):
    os.environ.setdefault(k, v)
import mlflow  # noqa: E402

LAB = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else LAB / "starter").resolve()
ok = bad = 0
out = []


def check(cid, cond, detail=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + detail if detail else ''}")


store = pathlib.Path(tempfile.mkdtemp(prefix="llmops_"))
env = {**os.environ, "LLMOPS_STORE": str(store), "PYTHONIOENCODING": "utf-8"}


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / "llmops.py"), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def current():
    f = store / "current_candidate.json"
    return json.loads(f.read_text(encoding="utf-8")).get("candidate") if f.exists() else None


def history():
    f = store / "promotion_log.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


for name in ("v1_base", "v1b_masked", "v2_guarded"):
    run("record", name)
mlflow.set_tracking_uri(f"sqlite:///{(store / 'mlflow.db').as_posix()}")
try:
    runs = mlflow.search_runs(experiment_names=["결석위험요약_후보"])
except Exception:
    runs = None
need = ["params.candidate", "params.prompt_sha", "params.code_sha", "params.evalset_sha", "params.model"]
have = runs is not None and len(runs) == 3 and all(c in runs.columns and runs[c].notna().all() for c in need)
check("G1  후보 셋의 기록에 이름·지시문·코드·평가 세트·모델이 다 남았다", have,
      "" if have else f"없는 칸 {[c.split('.')[1] for c in need if runs is None or c not in runs.columns]}")
if have:
    by = runs.set_index("params.candidate")
    check("G2  코드가 다른 후보는 지문이 다르고, 평가 세트의 지문은 셋이 같다",
          by["params.code_sha"].nunique() == 3 and by["params.evalset_sha"].nunique() == 1 and
          by.loc["v1_base", "params.prompt_sha"] == by.loc["v1b_masked", "params.prompt_sha"] != by.loc["v2_guarded", "params.prompt_sha"])
else:
    check("G2  코드가 다른 후보는 지문이 다르고, 평가 세트의 지문은 셋이 같다", False)

r1 = run("promote", "v1b_masked", "누출을 막았다")
check("G3  누출은 막았지만 사례 37건을 못 넘긴 후보는 거부된다 (종료코드 ≠ 0 · 운영 후보 없음)", r1.returncode != 0 and current() is None,
      f"종료코드 {r1.returncode} · 운영 후보 {current()}")
r2 = run("promote", "v2_guarded", "방어 코드를 더했다")
check("G4  전부 통과한 후보는 올라간다", r2.returncode == 0 and current() == "v2_guarded", f"종료코드 {r2.returncode} · 운영 후보 {current()}")
last = next((x for x in reversed(history()) if x.get("action") == "promote"), {})
keys = [k for k in ("from", "to", "why", "run_id", "prompt_sha", "evalset_sha") if k not in last]
check("G5  이력 한 줄에 무엇을·왜·어느 기록·어느 평가 세트인지가 있다", bool(last) and not keys, f"없는 칸 {keys}" if keys else "")

# 평가 세트를 한 글자 바꾼다 — 그때의 성적으로는 더 올릴 수 없어야 한다
golden = LAB / "eval_set" / "golden_cases.json"
backup = golden.read_bytes()
try:
    golden.write_bytes(backup + b"\n")
    (store / "current_candidate.json").unlink(missing_ok=True)
    r3 = run("promote", "v2_guarded", "평가 세트가 바뀐 뒤")
    check("G6  평가 세트가 바뀌면 옛 성적으로는 못 올린다", r3.returncode != 0 and current() is None, f"종료코드 {r3.returncode}")
finally:
    golden.write_bytes(backup)
shutil.rmtree(store, ignore_errors=True)

print("\n" + "\n".join(out) + f"\n\n통과 {ok} · 실패 {bad}\n")
sys.exit(1 if bad else 0)
