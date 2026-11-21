"""학습·서빙 어긋남 점검.  사용법:  python 심화/checks/check-skew.py [작업폴더]     (lab2_train_pipeline 폴더에서)

작업폴더(기본 시작코드)의 train.py 를 임시 폴더에 두 번 돌려 봅니다.
무엇을 보는가 — **학습할 때 한 손질이 모델과 함께 다니는가.**
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LAB = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else LAB / "시작코드").resolve()
SRC = LAB.parent / "공용데이터"
FEATURES = ["attend_rate_4w", "late_count_4w", "assign_submit_rate", "assign_late_rate", "consult_count", "prior_experience"]
ok = bad = 0
out = []


def check(cid, cond, detail=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + detail if detail else ''}")


def run(dest):
    return subprocess.run([sys.executable, str(ROOT / "train.py"), str(dest)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})


with tempfile.TemporaryDirectory() as tmp:
    a, b = pathlib.Path(tmp) / "a", pathlib.Path(tmp) / "b"
    ra, rb = run(a), run(b)
    model = joblib.load(a / "model.joblib") if (a / "model.joblib").exists() else None
    rec = json.loads((a / "RUN.json").read_text(encoding="utf-8")) if (a / "RUN.json").exists() else {}
    same = (a / "predictions.csv").exists() and (b / "predictions.csv").exists() and \
        (a / "predictions.csv").read_bytes() == (b / "predictions.csv").read_bytes()

check("S1  모델 파일(model.joblib)이 남는다", model is not None, (ra.stderr or "")[-200:])

raw = pd.read_csv(SRC / "trainees_2026-1.csv")  # 빈 값이 든 그대로 — 답하는 자리에 들어올 모양
try:
    proba = model.predict_proba(raw[FEATURES])[:, 1]
    fine = len(proba) == len(raw) and np.isfinite(proba).all()
    why = ""
except Exception as e:  # 채움이 모델 밖에 있으면 여기서 터진다
    fine, why = False, f"{type(e).__name__}: {str(e)[:80]}"
check("S2  빈 값이 든 입력을 모델이 그대로 받아 답한다", fine, why)

src = (ROOT / "train.py").read_text(encoding="utf-8")
code = "\n".join(line.split("#")[0] for line in src.splitlines())
check("S3  나누기 전에 전체 자료로 채우지 않는다 (fillna 가 없다)", not re.search(r"\.fillna\(", code))

try:
    files = [SRC / f"trainees_{c}.csv" for c in rec.get("cohorts", ["2025-1", "2025-2"])]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates()
    X_tr, _, _, _ = train_test_split(df[FEATURES], df["dropped"], test_size=rec["params"]["test_size"],
                                     random_state=rec["seed"], stratify=df["dropped"])
    imputer = next(step for _, step in model.steps if hasattr(step, "statistics_"))
    match = np.allclose(imputer.statistics_, X_tr.median().to_numpy(), atol=1e-9)
    why = "" if match else f"모델 안 {np.round(imputer.statistics_, 4).tolist()} · 학습셋 {X_tr.median().round(4).tolist()}"
except StopIteration:
    match, why = False, "모델 안에 채우는 칸이 없습니다"
except Exception as e:
    match, why = False, f"{type(e).__name__}: {str(e)[:80]}"
check("S4  채우는 값이 학습셋에서만 나왔다", match, why)

env = rec.get("env", {})
need = [k for k in ("python", "scikit-learn", "pandas") if not env.get(k)]
check("S5  RUN.json 에 돌린 환경(env)이 있다", not need, f"없는 것 {need}" if need else str(env))
check("S6  그래도 두 번 돌리면 예측이 같다", same, (rb.stderr or "")[-200:] if not same else "")

print("\n" + "\n".join(out) + f"\n\n통과 {ok} · 실패 {bad}\n")
sys.exit(1 if bad else 0)
