"""「그 차이는 우연인가」 점검.  사용법:  python 심화/checks/check-variance.py [작업폴더]     (lab3_experiment_registry 폴더에서)

작업폴더(기본 심화/starter)의 variance.json 과 variance.md 를 봅니다. 먼저 `python variance.py` 를 돌리십시오.
"""
import json
import pathlib
import re
import statistics
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAB = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else LAB / "심화" / "starter").resolve()
SRC = LAB.parent / "공용데이터"
ALL = ["attend_rate_4w", "late_count_4w", "assign_submit_rate", "assign_late_rate", "consult_count", "prior_experience"]
ok = bad = 0
out = []


def check(cid, cond, detail=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + detail if detail else ''}")


f = ROOT / "variance.json"
res = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
seeds = res.get("seeds", [])
rec, bc, th = res.get("recipes", {}), res.get("B_minus_C", {}), res.get("threshold", {})

check("V1  씨앗을 스무 개 이상 썼다", len(set(seeds)) >= 20, f"{len(set(seeds))}개")
keys = ("recall_mean", "recall_std", "precision_mean", "precision_std")
full = [n for n in ("A_기본", "B_가중치", "C_출석만") if all(isinstance(rec.get(n, {}).get(k), (int, float)) for k in keys)]
check("V2  설정 셋의 평균과 표준편차가 있다", len(full) == 3 and all(rec[n]["recall_std"] > 0 for n in full), f"채운 설정 {full}")

wins = [bc.get(k) for k in ("b_wins", "ties", "c_wins")]
check("V3  B 와 C 를 짝지어 셌다 (이김·비김·짐의 합 = 씨앗 수)",
      all(isinstance(w, int) for w in wins) and sum(wins) == len(seeds) > 0, f"{wins}")

try:  # 같은 씨앗으로 다시 재어 본다 — 적어 넣은 숫자가 아니라 잰 숫자인가
    df = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]], ignore_index=True).drop_duplicates()
    df[ALL] = df[ALL].fillna(df[ALL].median())
    vals = []
    for s in seeds:
        X_tr, X_va, y_tr, y_va = train_test_split(df[ALL], df["dropped"], test_size=0.25, random_state=s, stratify=df["dropped"])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20261121))
        vals.append(recall_score(y_va, m.fit(X_tr, y_tr).predict(X_va)))
    mine = round(statistics.mean(vals), 4)
    again = abs(mine - rec["B_가중치"]["recall_mean"]) < 0.002
    why = f"적힌 값 {rec['B_가중치']['recall_mean']} · 다시 잰 값 {mine}"
except Exception as e:
    again, why = False, f"{type(e).__name__}: {str(e)[:80]}"
check("V4  B 의 재현율 평균이 다시 재어도 같다", again, why)

num = all(isinstance(th.get(k), (int, float)) for k in ("best_threshold", "cost_at_best", "cost_at_0.5", "recall_at_best", "precision_at_best"))
check("V5  비용이 가장 작은 선을 찾았다 (0.5 일 때보다 비싸지 않다)",
      num and 0 < th["best_threshold"] < 1 and th["cost_at_best"] <= th["cost_at_0.5"],
      f"선 {th.get('best_threshold')} · 비용 {th.get('cost_at_best')} (0.5 일 때 {th.get('cost_at_0.5')})")

md = (ROOT / "variance.md").read_text(encoding="utf-8") if (ROOT / "variance.md").exists() else ""
m = re.search(r"^결론:[ \t]*(\S.*)$", md, re.M)
check("V6  variance.md 에 결론을 문장으로 적었다", bool(m) and len(m.group(1).strip()) >= 15)

print("\n" + "\n".join(out) + f"\n\n통과 {ok} · 실패 {bad}\n")
sys.exit(1 if bad else 0)
