"""데이터 계약·누수 점검.  사용법:  python 심화/checks/check-contract.py [작업폴더]     (lab1_data_version 폴더에서)

작업폴더(기본 심화/starter)에 contract.py · leakage_auc.py · leakage.md 가 있어야 합니다.
v1·v2 는 자기 것이 있으면 그것을(시작코드/data), 없으면 완성본/data 를 씁니다.
"""
import importlib.util
import pathlib
import re
import subprocess
import sys

import pandas as pd

LAB = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else LAB / "심화" / "starter").resolve()
DATA = next((d for d in (LAB / "시작코드" / "data", LAB / "완성본" / "data") if (d / "v2" / "trainees.csv").exists()), None)
ok = bad = 0
out = []


def check(cid, cond, detail=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + detail if detail else ''}")


spec = importlib.util.spec_from_file_location("contract_mod", ROOT / "contract.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
COLS = ["trainee_id", "cohort", "attend_rate_4w", "late_count_4w", "assign_submit_rate",
        "assign_late_rate", "consult_count", "prior_experience", "dropped"]

missing = [c for c in COLS if c not in mod.CONTRACT]
check("K1  계약에 아홉 칸이 다 있다", not missing, f"없는 칸 {missing}" if missing else "")

if DATA is None:
    check("K2  v1 에서 위반을 찾는다", False, "data/v1·v2 가 없습니다. 실습 1 을 먼저 끝내십시오")
    check("K3  v2 는 계약을 지킨다", False, "")
    v2 = None
else:
    v1, v2 = pd.read_csv(DATA / "v1" / "trainees.csv"), pd.read_csv(DATA / "v2" / "trainees.csv")
    f1, f2 = mod.violations(v1), mod.violations(v2)
    check("K2  v1 에서 빈 값과 겹친 번호를 둘 다 찾는다",
          any("빈" in x or "결측" in x for x in f1) and any("trainee_id" in x for x in f1), f"찾은 것 {f1}")
    check("K3  v2 는 계약을 지킨다 (위반 0)", not f2, f"남은 위반 {f2[:3]}" if f2 else "")

if v2 is not None:
    broken = v2.copy()
    broken.loc[broken.index[0], "attend_rate_4w"] = 1.3
    broken.loc[broken.index[1], "dropped"] = 2
    broken.loc[broken.index[2], "late_count_4w"] = -1
    found = [x for x in mod.violations(broken) if x not in f2]  # 일부러 깬 뒤에 **새로** 생긴 위반만
    hit = [c for c in ("attend_rate_4w", "dropped", "late_count_4w") if any(x.startswith(c) for x in found)]
    check("K4  범위 밖 출석률 · 라벨 2 · 음수 횟수를 셋 다 잡는다", len(hit) == 3, f"잡은 것 {hit}")
    extra = v2.assign(memo="x")
    check("K5  계약에 없는 칸이 끼어 오면 알린다", any("memo" in x for x in mod.violations(extra)))
    r = subprocess.run([sys.executable, str(ROOT / "contract.py"), str(DATA / "v1" / "trainees.csv")], capture_output=True)
    check("K6  어기면 종료코드가 0 이 아니다", r.returncode != 0, f"종료코드 {r.returncode}")
else:
    for cid in ("K4", "K5", "K6"):
        check(cid, False, "v2 가 없어 보지 못했습니다")

src = (ROOT / "leakage_auc.py").read_text(encoding="utf-8") if (ROOT / "leakage_auc.py").exists() else ""
m = re.search(r"^SAFE\s*=\s*\[(.*?)\]", src, re.M | re.S)
safe = re.findall(r'"(\w+)"', m.group(1)) if m else None
check("K7  넣어도 되는 칸에서 누수 칸 둘을 뺐다", safe is not None and "refund_requested" not in safe and "final_survey_done" not in safe,
      f"SAFE = {safe}")
md = (ROOT / "leakage.md").read_text(encoding="utf-8") if (ROOT / "leakage.md").exists() else ""
filled = len(re.findall(r"\|\s*(?:원래 6칸|6칸 \+[^|]*?)\s*\|\s*0\.\d+", md))
check("K8  leakage.md 에 AUC 셋을 옮겨 적었다", filled == 3, f"적은 줄 {filled}/3")

print("\n" + "\n".join(out) + f"\n\n통과 {ok} · 실패 {bad}\n")
sys.exit(1 if bad else 0)
