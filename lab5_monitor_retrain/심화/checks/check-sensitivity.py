"""「그 경보는 믿을 만한가」 점검.  사용법:  python 심화/checks/check-sensitivity.py [작업폴더]     (lab5_monitor_retrain 폴더에서)

작업폴더(기본 심화/starter)의 sensitivity.json 과 sensitivity.md 를 봅니다. 먼저 `python sensitivity.py` 를 돌리십시오.
"""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

LAB = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else LAB / "심화" / "starter").resolve()
SRC = LAB.parent / "공용데이터"
F = ["attend_rate_4w", "late_count_4w", "assign_submit_rate", "assign_late_rate", "consult_count", "prior_experience"]
ok = bad = 0
out = []


def check(cid, cond, detail=""):
    global ok, bad
    ok, bad = ok + bool(cond), bad + (not cond)
    out.append(f"{'  통과' if cond else '❌ 실패'}  {cid}{'  — ' + detail if detail else ''}")


def psi(base, new, bins):
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pb = np.clip(np.histogram(base, bins=edges)[0] / len(base), 1e-4, None)
    pn = np.clip(np.histogram(new, bins=edges)[0] / len(new), 1e-4, None)
    return float(((pn - pb) * np.log(pn / pb)).sum())


f = ROOT / "sensitivity.json"
res = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
bins, fa, ks = res.get("bins", {}), res.get("false_alarm", {}), res.get("ks", {})

full = all(isinstance(bins.get(c, {}).get(x, {}).get(b), (int, float)) for c in ("2026-1", "2026-2") for x in F for b in ("5", "10", "20"))
check("N1  두 기수 × 여섯 칸 × 구간 셋을 다 쟀다", full)

try:
    base = pd.concat([pd.read_csv(SRC / f"trainees_{c}.csv") for c in ["2025-1", "2025-2"]], ignore_index=True).drop_duplicates()
    new = pd.read_csv(SRC / "trainees_2026-2.csv")
    col = "assign_submit_rate"
    mine = {str(b): round(psi(base[col].to_numpy(float), new[col].to_numpy(float), b), 4) for b in (5, 10, 20)}
    same = all(abs(mine[b] - bins["2026-2"][col][b]) < 0.002 for b in mine)
    why = f"적힌 값 {bins['2026-2'][col]} · 다시 잰 값 {mine}"
except Exception as e:
    same, why = False, f"{type(e).__name__}: {str(e)[:80]}"
check("N2  과제 제출률의 PSI 가 다시 재어도 같다", same, why)

sizes = [s for s in ("310", "100", "50") if all(isinstance(fa.get(s, {}).get(k), (int, float)) and 0 <= fa[s][k] <= 1 for k in ("any_watch", "any_alert"))]
check("N3  헛경보를 표본 수 셋에서 200번 이상 쟀다", len(sizes) == 3 and res.get("repeats", 0) >= 200, f"채운 표본 수 {sizes} · 반복 {res.get('repeats')}")
check("N4  표본이 작을수록 헛경보가 는다 (50 > 310)",
      len(sizes) == 3 and fa["50"]["any_alert"] > fa["310"]["any_alert"] and fa["50"]["any_watch"] >= fa["310"]["any_watch"],
      f"이상 비율 310:{fa.get('310', {}).get('any_alert')} · 50:{fa.get('50', {}).get('any_alert')}" if len(sizes) == 3 else "")

try:
    kfine = all(isinstance(ks[c][x], (int, float)) for c in ("2026-1", "2026-2") for x in F) and \
        ks["2026-2"]["attend_rate_4w"] < 0.01 and min(ks["2026-1"].values()) > 0.05
    why = f"2026-1 가장 작은 p {min(ks['2026-1'].values())} · 2026-2 출석률 p {ks['2026-2']['attend_rate_4w']}"
except Exception as e:
    kfine, why = False, "ks 가 비었거나 모양이 다릅니다"
check("N5  KS 검정이 바뀐 세상은 잡고 같은 세상은 안 잡는다", kfine, why)

md = (ROOT / "sensitivity.md").read_text(encoding="utf-8") if (ROOT / "sensitivity.md").exists() else ""
m = re.search(r"^결론:[ \t]*(\S.*)$", md, re.M)
check("N6  sensitivity.md 에 결론을 문장으로 적었다", bool(m) and len(m.group(1).strip()) >= 15)

print("\n" + "\n".join(out) + f"\n\n통과 {ok} · 실패 {bad}\n")
sys.exit(1 if bad else 0)
